"""Общие настройки эксперимента: пути, идентификаторы моделей, параметры по умолчанию."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # сюда скачивается DIV2K (в .gitignore)
CKPT_DIR = ROOT / "checkpoints"     # сюда скачиваются веса FaithDiff (в .gitignore)
RESULTS_DIR = ROOT / "results"      # картинки и metrics.csv (в .gitignore)
REPORT_DIR = ROOT / "report"        # графики и таблицы для README (коммитятся)

# --- Модель -----------------------------------------------------------------
# FaithDiff (CVPR 2025) поверх SDXL. Базовый чекпоинт SDXL тот же, что
# в официальном примере diffusers.
SDXL_MODEL_ID = "SG161222/RealVisXL_V4.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
FAITHDIFF_REPO = "jychen9811/FaithDiff"
FAITHDIFF_FILE = "FaithDiff.bin"
COMMUNITY_PIPELINE = "pipeline_faithdiff_stable_diffusion_xl"
DIFFUSERS_REVISION = "0.42.1"       # версия community-пайплайна = версия diffusers

# --- Сэмплинг (значения по умолчанию из test_wo_llava.py авторов) -------------
UPSCALE = 4
NUM_STEPS = 20
GUIDANCE_SCALE = 5.0
START_POINT = "lr"                  # 'lr' или 'noise'
TILE_SIZE = 1024                    # размер тайла в пикселях (латент = /8)
TILE_OVERLAP = 0.5
SEED = 42
PROMPT = ("A high quality, highly detailed, sharp photograph with realistic "
          "textures and natural colors.")
NEGATIVE_PROMPT = ("blurry, low quality, jpeg artifacts, noise, oversmoothed, "
                   "painting, cartoon, deformed, watermark")

# --- Маска ------------------------------------------------------------------
LATENT_DILATE = 1      # на сколько латентных пикселей расширять маску (шов генерируется моделью)
FEATHER_PX = 8         # sigma гауссова размытия маски при финальной склейке в пикселях
BLEND_END = 1.0        # доля шагов, на которых выполняется замена известной области в латенте

# --- Данные -----------------------------------------------------------------
DIV2K_URLS = {
    "HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "bicubic": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
    "unknown": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip",
}
IMAGE_IDS = ["0801", "0804", "0811", "0829", "0853"]
DEGRADATIONS = ["bicubic", "unknown", "realesrgan"]
MASKS = ["ones", "zeros", "combined"]
LR_CROP = 256          # LR-кроп 256x256 -> выход 1024x1024 = ровно один тайл SDXL
