import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "sentiment.db"

load_dotenv(ROOT / ".env")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "SentimentSignals/1.0 maanitkeyhem@gmail.com",
)

DATA_DIR.mkdir(exist_ok=True)
