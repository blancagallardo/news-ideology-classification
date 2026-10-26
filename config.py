from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv

AÑOS = 3
FECHA_LIMITE = datetime.now() - timedelta(days=365 * AÑOS)

DELAY_MIN = 2.0
DELAY_MAX = 5.0
DELAY_AFTER_ERROR = 10.0   # pausa extra tras error


TIMEOUT = 20
MAX_RETRIES = 3

# ── Volumen ──────────────────────────────────────────────────
MAX_PAGINAS_POR_SECCION = 500   # para paginación HTML
MAX_ARTICULOS_POR_FEED  = 9999 # sin límite efectivo en RSS

# ── Archivos ─────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"  # un .json por medio
OUTPUT_DIR     = "output"  # JSONs finales por medio
LOG_FILE       = "logs/scraper.log"

# ── Temas a recopilar ────────────────────────────────────────
TEMAS = ["politica", "economia", "opinion"]

# ── User-Agents rotativos ────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


# Carga el .env desde la raíz del proyecto
load_dotenv(Path(__file__).parent / ".env")

# ── API Keys ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")

# Validación al importar
_missing = [k for k, v in {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
}.items() if not v]

if _missing:
    raise EnvironmentError(
        f"Faltan las siguientes keys en el .env: {', '.join(_missing)}\n"
        "Crea el fichero .env en la raíz del proyecto con el formato:\n"
        "  ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE\n"
        "  OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE\n"
        "  GROQ_API_KEY=gsk_YOUR_KEY_HERE"
    )