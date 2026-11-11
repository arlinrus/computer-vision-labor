"""SR-inpainting для FaithDiff (CVPR 2025).

Загрузка FaithDiff через community-пайплайн diffusers и ПЕРЕПИСАННЫЙ цикл
сэмплинга, в котором улучшается только область, заданная маской.

Идея (Blended Latent Diffusion / RePaint, адаптированные под SR):
  * "известная" область — латент z_ref бикубически увеличенного входа;
  * после каждого шага решателя x_{t-1} в латенте склеивается:
        x_{t-1} <- m * x_{t-1}^model + (1 - m) * q(z_ref, t-1),
    где q(z_ref, t) = sqrt(abar_t) z_ref + sqrt(1 - abar_t) eps — прямой процесс;
  * тайлы, полностью лежащие вне маски, UNet не считает вовсе (экономия времени);
  * после декодирования VAE финальная склейка в пикселях с размытой маской,
    чтобы область вне маски совпадала со входом бит-в-бит (VAE не идеален).
"""
import copy
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from masks import latent_mask, pixel_mask


# --------------------------------------------------------------------------
# Загрузка модели
# --------------------------------------------------------------------------
def load_pipeline(dtype=torch.float16, cpu_offload: bool = False, scheduler: str = "unipc"):
    from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, UniPCMultistepScheduler
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(config.FAITHDIFF_REPO, filename=config.FAITHDIFF_FILE,
                              local_dir=str(config.CKPT_DIR / "faithdiff"))
    vae = AutoencoderKL.from_pretrained(config.VAE_MODEL_ID, torch_dtype=dtype)
    pipe = DiffusionPipeline.from_pretrained(
        config.SDXL_MODEL_ID,
        torch_dtype=dtype,
        vae=vae,
        unet=None,  # UNet создаётся классом из community-пайплайна
        custom_pipeline=config.COMMUNITY_PIPELINE,
        custom_revision=config.DIFFUSERS_REVISION,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.unet = pipe.unet_model.from_pretrained(config.SDXL_MODEL_ID, subfolder="unet",
                                                variant="fp16", use_safetensors=True)
    pipe.unet.load_additional_layers(weight_path=weights, dtype=dtype)
    pipe.set_encoder_tile_settings()
    pipe.vae.enable_tiling()
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    # Склейка с q(z_ref, t) корректна для VP-расписаний (eps/v-prediction без
    # масштабирования sigma), поэтому поддерживаем только UniPC (как у авторов) и DDIM.
    sched_cls = {"unipc": UniPCMultistepScheduler, "ddim": DDIMScheduler}[scheduler]
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
    return pipe


def _module_fn(pipe, name):
    """Функции из модуля community-пайплайна (retrieve_timesteps, LocalAttention, ...)."""
    return getattr(sys.modules[type(pipe).__module__], name)


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> [1,3,H,W] float в [0,1]."""
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)[None]


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((x * 255).round().astype(np.uint8))


def adain_color_fix(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Перенос поканальных mean/std (как adain_color_fix у авторов FaithDiff)."""
    t_mean, t_std = target.mean((2, 3), keepdim=True), target.std((2, 3), keepdim=True) + 1e-6
    s_mean, s_std = source.mean((2, 3), keepdim=True), source.std((2, 3), keepdim=True)
    return ((target - t_mean) / t_std * s_std + s_mean).clamp(0, 1)


def _pad_to_multiple(img: Image.Image, mult: int = 8):
    w, h = img.size
    ph, pw = (mult - h % mult) % mult, (mult - w % mult) % mult
    arr = np.pad(np.asarray(img), ((0, ph), (0, pw), (0, 0)), mode="edge")
    return Image.fromarray(arr), (h, w)


# --------------------------------------------------------------------------
# Основной алгоритм
# --------------------------------------------------------------------------
@torch.no_grad()
def masked_super_resolve(
    pipe,
    lr_image: Image.Image,
    mask_lr: np.ndarray,
    upscale: int = config.UPSCALE,
    prompt: str = config.PROMPT,
    negative_prompt: str = config.NEGATIVE_PROMPT,
    num_inference_steps: int = config.NUM_STEPS,
    guidance_scale: float = config.GUIDANCE_SCALE,
    start_point: str = config.START_POINT,
    tile_size: int = config.TILE_SIZE,
    overlap: float = config.TILE_OVERLAP,
    latent_dilate: int = config.LATENT_DILATE,
    feather_px: float = config.FEATHER_PX,
    blend_end: float = config.BLEND_END,
    color_fix: bool = True,
    seed: int = config.SEED,
):
    """Улучшает только область mask_lr==1.

    lr_image : PIL RGB, входное изображение h x w
    mask_lr  : бинарный массив h x w (разрешение входа)
    Возвращает (PIL результат h*upscale x w*upscale, словарь статистик).
    """
    assert mask_lr.shape == (lr_image.height, lr_image.width), "маска должна совпадать со входом"
    device = torch.device("cuda")  # community-пайплайн жёстко использует CUDA
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    # --- 0. Бикубический апскейл входа и маски до целевого разрешения ---------
    W, H = lr_image.width * upscale, lr_image.height * upscale
    lq_up = lr_image.resize((W, H), Image.BICUBIC)
    lq_up_pad, (H0, W0) = _pad_to_multiple(lq_up, 8)
    Hp, Wp = lq_up_pad.height, lq_up_pad.width
    mask_hr = torch.from_numpy(mask_lr.astype(np.float32))[None, None]
    mask_hr = F.interpolate(mask_hr, size=(H, W), mode="nearest")
    mask_hr = F.pad(mask_hr, (0, Wp - W, 0, Hp - H), mode="replicate").to(device)

    lq_up_t = pil_to_tensor(lq_up).to(device)
    coverage = float(mask_hr[..., :H, :W].mean())
    stats = {"coverage": coverage, "unet_tiles": 0, "skipped_tiles": 0}

    # Тривиальный случай: улучшать нечего -> результат = бикубический апскейл.
    if coverage == 0.0:
        stats.update(time_s=time.time() - t0, peak_mem_gb=0.0)
        return lq_up, stats

    retrieve_timesteps = _module_fn(pipe, "retrieve_timesteps")
    LocalAttention = _module_fn(pipe, "LocalAttention")
    dtype = pipe.unet.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    target_size = (tile_size, tile_size)
    original_size = (Hp, Wp)
    kernel = (tile_size // 8, tile_size // 8)

    # Отдельные "тайлеры" для латентов, условия, маски и известной области:
    # LocalAttention хранит индексы последнего разбиения внутри себя.
    tlc_lat = LocalAttention(kernel, overlap)
    tlc_img = LocalAttention(kernel, overlap)
    tlc_aux = LocalAttention(kernel, overlap)

    # --- 1. Текст ------------------------------------------------------------
    prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
        prompt, num_images_per_prompt=1, do_classifier_free_guidance=True,
        negative_prompt=negative_prompt)
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], 0).to(device)
        add_text_embeds = torch.cat([neg_pooled, pooled], 0).to(device)
    else:
        prompt_embeds, add_text_embeds = prompt_embeds.to(device), pooled.to(device)

    # --- 2. Условие FaithDiff (выровненные признаки LQ) -----------------------
    lr_img = pipe.image_processor.preprocess([lq_up_pad], height=Hp, width=Wp).to(device, dtype=dtype)
    image_latents = pipe.prepare_image_latents(lr_img, 1, 1, dtype, device, do_cfg)
    image_latents = tlc_img.grids(image_latents)

    # --- 3. Латент известной области z_ref и маска в латенте ------------------
    z_ref = pipe.vae.encode(lr_img.to(pipe.vae.dtype)).latent_dist.mode()
    z_ref = (z_ref * pipe.vae.config.scaling_factor).to(dtype)
    m_lat = latent_mask(mask_hr, pipe.vae_scale_factor, latent_dilate).to(device, dtype)

    # --- 4. Расписание и начальный шум ----------------------------------------
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, None)
    noise = torch.randn(z_ref.shape, generator=generator, device=device, dtype=dtype)
    ddpm = pipe.DDPMScheduler  # тот же beta-schedule, используем только add_noise

    def q_sample(t: int) -> torch.Tensor:
        """Известная область, зашумлённая до уровня t (тот же eps, что и в старте)."""
        if t <= 0:
            return z_ref
        return ddpm.add_noise(z_ref, noise, torch.tensor([int(t)], device=device).long())

    if start_point == "lr":
        latents = q_sample(int(timesteps[0]))
    else:
        latents = noise * pipe.scheduler.init_noise_sigma
    latents = tlc_lat.grids(latents)
    n_tiles = latents.shape[0]

    # Какие тайлы вообще содержат маску: остальные UNet не считает.
    mask_tiles = tlc_aux.grids(m_lat)
    active = [bool(mask_tiles[k].max() > 0) for k in range(n_tiles)]
    stats["skipped_tiles"] = (len(active) - sum(active)) * len(timesteps)

    sched_state = [copy.deepcopy(pipe.scheduler.__dict__) for _ in range(n_tiles)]
    n_blend = int(round(blend_end * len(timesteps)))

    # --- 5. Цикл сэмплинга ------------------------------------------------------
    for i, t in enumerate(pipe.progress_bar(timesteps)):
        if i >= 1:
            latents = tlc_lat.grids(latents).to(dtype)
        t_prev = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
        ref_prev_tiles = tlc_aux.grids(q_sample(t_prev))

        out_tiles = []
        for k in range(n_tiles):
            if not active[k]:
                # Тайл целиком вне маски: сразу ставим известную область уровня t_prev.
                out_tiles.append(ref_prev_tiles[k:k + 1])
                continue
            pipe.scheduler.__dict__.update(sched_state[k])
            sub = latents[k:k + 1]
            cond = image_latents[k:k + 1]
            model_in = torch.cat([sub] * 2) if do_cfg else sub
            cond_in = torch.cat([cond] * 2) if do_cfg else cond
            model_in = pipe.scheduler.scale_model_input(model_in, t)
            pos = tlc_lat.idxes[k]
            add_time_ids = torch.tensor(
                [[*original_size, pos["i"], pos["j"], *target_size]], device=device, dtype=dtype
            ).repeat(model_in.shape[0], 1)
            noise_pred = pipe.unet(
                model_in, t,
                encoder_hidden_states=prompt_embeds,
                input_embedding=cond_in,
                add_sample=True,
                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                return_dict=False,
            )[0]
            if do_cfg:
                u, c = noise_pred.chunk(2)
                noise_pred = u + guidance_scale * (c - u)
            sub = pipe.scheduler.step(noise_pred, t, sub, return_dict=False)[0]
            sched_state[k] = copy.deepcopy(pipe.scheduler.__dict__)
            stats["unet_tiles"] += 1
            out_tiles.append(sub.to(dtype))

        latents = tlc_lat.grids_inverse(torch.cat(out_tiles, 0)).to(dtype)

        # === МОДИФИКАЦИЯ СЭМПЛИНГА: склейка с известной областью ===
        if i < n_blend:
            latents = m_lat * latents + (1.0 - m_lat) * q_sample(t_prev)

    # --- 6. Декодирование -------------------------------------------------------
    upcast = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if upcast:
        pipe.vae.to(dtype=torch.float32)
    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    if upcast:
        pipe.vae.to(dtype=torch.float16)
    sr = ((img.float() + 1) / 2).clamp(0, 1)[..., :H0, :W0]

    if color_fix:
        sr = adain_color_fix(sr, lq_up_t)

    # --- 7. Финальная склейка в пикселях --------------------------------------
    m_px = pixel_mask(mask_hr[..., :H0, :W0], feather_px)
    out = m_px * sr + (1.0 - m_px) * lq_up_t

    pipe.maybe_free_model_hooks()
    stats.update(time_s=time.time() - t0, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    return tensor_to_pil(out), stats
