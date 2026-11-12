import torch

import config

from diffusers import AutoencoderKL, DiffusionPipeline, UniPCMultistepScheduler
from huggingface_hub import hf_hub_download

def load_model(cpu_offload: bool = True):

    weights = hf_hub_download(config.FAITHDIFF_REPO, filename=config.FAITHDIFF_FILE)  # скачать FaithDiff.bin
    vae = AutoencoderKL.from_pretrained(config.VAE_ID, torch_dtype=torch.float16)     # скачать VAE

    pipe = DiffusionPipeline.from_pretrained(       # "коробка" со всеми деталями модели
        config.SDXL_ID,
        torch_dtype=torch.float16,                  # половинная точность -> меньше памяти
        vae=vae,                                    # наш VAE вместо родного
        unet=None,                                  # UNet загружаем отдельно ниже
        custom_pipeline="pipeline_faithdiff_stable_diffusion_xl",  # код FaithDiff из diffusers
        use_safetensors=True,
        variant="fp16",
    )
    pipe.unet = pipe.unet_model.from_pretrained(    # UNet нейросеть, угадывающая шум
        config.SDXL_ID, subfolder="unet", variant="fp16", use_safetensors=True)
    pipe.unet.load_additional_layers(weight_path=weights, dtype=torch.float16)  # + слои FaithDiff

    pipe.vae.enable_tiling()                        # VAE режет большие картинки на куски (экономия памяти)
    pipe.unet.denoise_encoder.enable_tiling()       # то же для энкодера плохой картинки
    if cpu_offload:
        pipe.enable_model_cpu_offload()             # части модели ждут в ОЗУ и по очереди едут на GPU
    else:
        pipe = pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)  
    return pipe
