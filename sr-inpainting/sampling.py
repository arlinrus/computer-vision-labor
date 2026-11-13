import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
 
def to_pil(x: torch.Tensor) -> Image.Image:
    x = x[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((x * 255).round().astype(np.uint8))

@torch.no_grad()
def masked_sr(pipe, lr: Image.Image, mask: np.ndarray) -> Image.Image:
    assert mask.shape == (lr.height, lr.width), "маска должна совпадать по размеру со входом"
    assert max(lr.size) <= 256 and lr.width % 2 == 0 and lr.height % 2 == 0, "вход до 256x256, чётные стороны"
    device, dtype = torch.device("cuda"), torch.float16
    W, H = lr.width * config.UPSCALE, lr.height * config.UPSCALE #размер выхода 256 на 4

    up = lr.resize((W, H), Image.BICUBIC)                       
    if mask.sum() == 0:                                         # маска пустая -> модель не нужна
        return up
    mask_hr = torch.from_numpy(mask.astype(np.float32))[None, None]       # [1, 1, 256, 256]
    mask_hr = F.interpolate(mask_hr, size=(H, W), mode="nearest").to(device)  # [1, 1, 1024, 1024], всё ещё 0/1

    pos, neg, pos_pooled, neg_pooled = pipe.encode_prompt(
        config.PROMPT, num_images_per_prompt=1, do_classifier_free_guidance=True,
        negative_prompt=config.NEGATIVE_PROMPT)
    text = torch.cat([neg, pos]).to(device)                     # 2 варианта: [плохой, хороший] для CFG
    text_pooled = torch.cat([neg_pooled, pos_pooled]).to(device)
    time_ids = torch.tensor([[H, W, 0, 0, 1024, 1024]], device=device, dtype=dtype).repeat(2, 1)  # размер/позиция тайла

    x = pipe.image_processor.preprocess([up], height=H, width=W).to(device, dtype)  # [-1..1]
    cond = pipe.prepare_image_latents(x, 1, 1, dtype, device, True)  # признаки от энкодера FaithDiff

    z_ref = pipe.vae.encode(x.to(pipe.vae.dtype)).latent_dist.mode() * pipe.vae.config.scaling_factor
    z_ref = z_ref.to(dtype)                                     # [1, 4, 128, 128] — «как было»
    m = F.max_pool2d(mask_hr, kernel_size=8).to(dtype)          # маска 1024 -> 128: латентный пиксель = 1,
                                                                # если в его квадрате 8x8 есть хоть одна единица

    pipe.scheduler.set_timesteps(config.STEPS, device=device)   # 20 шагов: t = 999, 949, ... (примерно)
    timesteps = pipe.scheduler.timesteps
    gen = torch.Generator(device=device).manual_seed(config.SEED)
    noise = torch.randn(z_ref.shape, generator=gen, device=device, dtype=dtype)

    def noisy_ref(t: int) -> torch.Tensor:
        if t <= 0:
            return z_ref
        return pipe.DDPMScheduler.add_noise(z_ref, noise, torch.tensor([t], device=device))

    latents = noisy_ref(999)                                    # start_point="lr": старт с зашумлённой картинки

    for i, t in enumerate(pipe.progress_bar(timesteps)):
        inp = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)   # x2 — для CFG
        eps = pipe.unet(
            inp, t,
            encoder_hidden_states=text,                         # текст
            input_embedding=torch.cat([cond] * 2),              # плохая картинка (то, что добавил FaithDiff)
            add_sample=True,
            added_cond_kwargs={"text_embeds": text_pooled, "time_ids": time_ids},
            return_dict=False,
        )[0]
        eps_bad, eps_good = eps.chunk(2)
        eps = eps_bad + config.GUIDANCE * (eps_good - eps_bad)   # CFG: уходим от плохого к хорошему

        latents = pipe.scheduler.step(eps, t, latents, return_dict=False)[0].to(dtype)

        t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
        latents = m * latents + (1 - m) * noisy_ref(t_next)

    img = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(pipe.vae.dtype), return_dict=False)[0]
    sr = ((img.float() + 1) / 2).clamp(0, 1)                    

    up_t = to_tensor(up).to(device)
    sr = (sr - sr.mean((2, 3), keepdim=True)) / (sr.std((2, 3), keepdim=True) + 1e-6) \
        * up_t.std((2, 3), keepdim=True) + up_t.mean((2, 3), keepdim=True)   

    k = 6 * config.FEATHER_PX + 1                               # размер ядра размытия
    soft = cv2.GaussianBlur(mask_hr[0, 0].cpu().numpy(), (k, k), config.FEATHER_PX)  # мягкий край маски
    soft = torch.from_numpy(soft)[None, None].to(device)
    out = soft * sr.clamp(0, 1) + (1 - soft) * up_t             # внутри маски модель, снаружи бикубика

    pipe.maybe_free_model_hooks()
    return to_pil(out)
