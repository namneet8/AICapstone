 from pathlib import Path
   
   BASE_DIR = Path(__file__).parent
   RAW_DATA = BASE_DIR / "data" / "raw"
   PROCESSED_DATA = BASE_DIR / "data" / "processed"
   EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
   MODELS_DIR = BASE_DIR / "models"