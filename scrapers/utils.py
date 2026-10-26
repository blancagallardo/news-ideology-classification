# utils.py — Funciones de soporte

import time
import random
import logging
import json
import os
import re
from datetime import datetime
from pathlib import Path
import config

log = logging.getLogger(__name__)


# ── HTTP ─────────────────────────────────────────────────────

def get_headers():
    return {
        "User-Agent": random.choice(config.USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "DNT": "1",
    }


def sleep_random():
    t = random.uniform(config.DELAY_MIN, config.DELAY_MAX)
    time.sleep(t)


def fetch(url, session):
    for attempt in range(config.MAX_RETRIES):
        try:
            sleep_random()
            r = session.get(url, headers=get_headers(), timeout=config.TIMEOUT)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r
        except Exception as e:
            log.warning(f"[{attempt+1}/{config.MAX_RETRIES}] Error en {url}: {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.DELAY_AFTER_ERROR)
    log.error(f"Fallido definitivamente: {url}")
    return None


# ── Fechas ───────────────────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%Y-%m-%d",
]

def parse_date(date_str):
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=None).isoformat()
        except:
            pass
    # Intentar fromisoformat como último recurso
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None).isoformat()
    except:
        return date_str


def is_within_range(date_iso):
    """True si la fecha está dentro del rango configurado."""
    if not date_iso:
        return True
    try:
        dt = datetime.fromisoformat(str(date_iso).split("+")[0].split("Z")[0])
        return dt >= config.FECHA_LIMITE
    except:
        return True  # si no se puede parsear, incluir


def date_is_too_old(date_iso):
    """True si ya pasamos el límite temporal (para detener paginación)."""
    if not date_iso:
        return False
    try:
        dt = datetime.fromisoformat(str(date_iso).split("+")[0].split("Z")[0])
        return dt < config.FECHA_LIMITE
    except:
        return False


# ── Texto ────────────────────────────────────────────────────

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text)).strip()
    # Eliminar caracteres de control
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def extract_body(soup, selectors):
    """Prueba una lista de selectores CSS/clase hasta encontrar el cuerpo."""
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            paragraphs = tag.find_all("p")
            if paragraphs:
                return clean_text(" ".join(p.get_text() for p in paragraphs))
            # Si no hay <p>, devolver texto directo
            text = clean_text(tag.get_text())
            if len(text) > 100:
                return text
    return ""


_JLD_TYPES = {"NewsArticle", "Article", "ReportageNewsArticle", "OpinionNewsArticle"}

def _jld_type_matches(data):
    """Comprueba @type tanto si es string como si es lista."""
    t = data.get("@type", "")
    if isinstance(t, list):
        return any(x in _JLD_TYPES for x in t)
    return t in _JLD_TYPES

def extract_json_ld(soup):
    """Extrae datos del script JSON-LD si existe."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = next((d for d in data if _jld_type_matches(d)), data[0] if data else {})
            if _jld_type_matches(data):
                return data
        except:
            pass
    return {}


# ── Clasificación de temas ───────────────────────────────────

KEYWORDS = {
    "politica": [
        "gobierno", "congreso", "partido", "elecciones", "ministro", "presidente",
        "psoe", "pp", "vox", "podemos", "sumar", "parlamento", "senado", "diputado",
        "sánchez", "feijóo", "legislatura", "moción", "debate", "constitución",
        "comunidad autónoma", "generalitat", "junta", "autonomía"
    ],
    "economia": [
        "pib", "inflación", "desempleo", "paro", "bolsa", "mercado", "banco",
        "deuda", "presupuesto", "iva", "irpf", "euríbor", "hipoteca", "precio",
        "salario", "sueldo", "empresa", "exportación", "importación", "inversión",
        "déficit", "pensión", "banco de españa", "bce", "fed", "economía",
        "recesión", "crecimiento", "empleo", "trabajo"
    ],
    "opinion": [
        "editorial", "tribuna", "columna", "análisis", "opinión", "perspectiva",
        "reflexión", "debate", "carta al director", "punto de vista", "comentario"
    ],
}

def clasificar_tema(titular, abstract, cuerpo=""):
    texto = f"{titular} {abstract} {cuerpo[:300]}".lower()
    puntos = {tema: sum(1 for kw in kws if kw in texto)
              for tema, kws in KEYWORDS.items()}
    mejor = max(puntos, key=puntos.get)
    return mejor if puntos[mejor] > 0 else "politica"


# ── Checkpoint ───────────────────────────────────────────────

def load_checkpoint(medio):
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    path = f"{config.CHECKPOINT_DIR}/{medio}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"urls": [], "articulos": []}


def save_checkpoint(medio, data):
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    path = f"{config.CHECKPOINT_DIR}/{medio}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def save_output(medio, articulos):
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    path = f"{config.OUTPUT_DIR}/{medio}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articulos, f, ensure_ascii=False, indent=2)
    return path
