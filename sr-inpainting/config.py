from pathlib import Path

ROOT = Path(__file__).resolve().parent      
PHOTOS_DIR = ROOT / "my_photos"             
RESULTS_DIR = ROOT / "results"             
REPORT_DIR = ROOT / "report"               
SDXL_ID = "SG161222/RealVisXL_V4.0"       
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"   
FAITHDIFF_REPO = "jychen9811/FaithDiff"     
FAITHDIFF_FILE = "FaithDiff.bin"           

UPSCALE = 4                                
STEPS = 20                                 
GUIDANCE = 5.0                              
SEED = 42                                  
PROMPT = "A high quality, highly detailed, sharp photograph with realistic textures and natural colors."
NEGATIVE_PROMPT = "blurry, low quality, jpeg artifacts, noise, oversmoothed, painting, cartoon"

FEATHER_PX = 8                           

IMAGE_IDS = ["cat", "building", "moh"]                  
DEGRADATIONS = ["bicubic", "unknown", "realesrgan"]     
MASKS = ["ones", "zeros", "combined"]                  
LR_SIZE = 256                                          
