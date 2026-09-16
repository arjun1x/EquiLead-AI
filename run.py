"""Use python run.py from any working directory; defaults to localhost only."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8000")))
