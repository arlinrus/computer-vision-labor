from pathlib import Path

ROOT = Path(__file__).resolve().parent      
DATA_DIR = ROOT / "data"                    
RESULTS_DIR = ROOT / "results"             
REPORT_DIR = ROOT / "report"                

SDXL_ID = "SG161222/RealVisXL_V4.0"         # основная нейросеть SDXL, рисует фото
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"    # VAE: картинка <-> латент (сжатие в 8 раз)
FAITHDIFF_REPO = "jychen9811/FaithDiff"     # репозиторий с весами FaithDiff
FAITHDIFF_FILE = "FaithDiff.bin"            # слои, через которые модель видит плохую картинку

UPSCALE = 4                                 # увеличение: 256 -> 1024
STEPS = 20                                  # шагов очистки от шума
GUIDANCE = 5.0                              # сила промпта (CFG)
SEED = 42                                   # фиксированная случайность
PROMPT = "A high quality, highly detailed, sharp photograph with realistic textures and natural colors."
NEGATIVE_PROMPT = "blurry, low quality, jpeg artifacts, noise, oversmoothed, painting, cartoon"

FEATHER_PX = 8                              # насколько размыть край маски при финальной склейке

IMAGE_IDS = ["0801", "0804", "0811", "0829", "0853"]   # картинки DIV2K
DEGRADATIONS = ["bicubic", "unknown", "realesrgan"]     # 3 вида порчи
MASKS = ["ones", "zeros", "combined"]                   # 3 маски из задания
LR_SIZE = 256                                           # вход 256x256 -> выход 1024x1024
