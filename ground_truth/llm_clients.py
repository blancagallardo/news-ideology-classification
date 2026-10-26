"""
Árbitros LLM (Kennedy et al. 2025, Table 2):
    Claude Sonnet 4.6  F1=0.462  mejor clasificador comercial
    GPT-5.2            F1=0.448  segundo comercial
    Llama 3.1 8B       F1=0.397  mejor open-weight (Groq, gratis)

Sesgos conocidos:
    Claude y GPT: center-collapse sistemático
    Llama:        sesgo LEFT propio
"""

import json
import re
import sys
import time
from pathlib import Path

import anthropic
import openai
from groq import Groq

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY

from .voting import normalize_label

# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un analista político experto en detección de sesgo mediático en prensa española.
Tu tarea es clasificar la ideología política expresada en un artículo de noticias como IZQUIERDA, CENTRO o DERECHA.

REGLAS IMPORTANTES:
- Basa tu clasificación ÚNICAMENTE en el contenido lingüístico del artículo (elección de palabras, encuadre, omisiones, énfasis).
- NO asignes automáticamente la ideología del medio al artículo. Un artículo de un medio de izquierda puede ser CENTRO o DERECHA en su contenido real, y viceversa.
- Presta atención a señales ideológicas específicas del contexto español: terminología sobre inmigración, economía, identidad nacional, género, religión y relaciones laborales.
- Responde ÚNICAMENTE en formato JSON válido, sin texto adicional, sin bloques de código markdown."""


def build_prompt(titular: str, abstract: str, noticia: str) -> str:
    noticia_truncada = " ".join(noticia.split()[:400])

    # Evitar duplicar si abstract ya está en el inicio de la noticia
    abstract_section = ""
    if abstract and abstract.strip() and abstract.strip() not in noticia[:500]:
        abstract_section = f"Resumen: {abstract}\n"

    return f"""Analiza este artículo de noticias en español y clasifica su ideología política.

ARTÍCULO:
Titular: {titular}
{abstract_section}Cuerpo: {noticia_truncada}

TAREA:
1. Clasifica el artículo como IZQUIERDA, CENTRO o DERECHA basándote únicamente en su contenido lingüístico.
2. Justifica tu clasificación explicando los elementos del texto que la revelan.
3. Señala qué elementos textuales concretos revelan la ideología: léxico cargado, encuadre narrativo y omisiones relevantes.

Responde con esta estructura JSON exacta (sin texto adicional):
{{
  "label": "IZQUIERDA" | "CENTRO" | "DERECHA",
  "confidence": "ALTA" | "MEDIA" | "BAJA",
  "justificacion": "2-3 frases explicando por qué se asignó esta etiqueta ideológica",
  "evidencia_textual": {{
    "lexico": ["palabra_cargada_1", "palabra_cargada_2"],
    "encuadre": "descripción breve de cómo se enmarca el tema ideológicamente",
    "omisiones": "qué perspectivas, voces o datos faltan o se minimizan"
  }}
}}"""


# ─────────────────────────────────────────────────────────────
# UTILIDADES DE PARSEO
# ─────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    candidate = match.group() if match else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: extraer campos clave con regex cuando el JSON viene malformado
    # (p.ej. comillas sin escapar en strings de justificación)
    result = {}
    for field, pattern in [
        ("label",      r'"label"\s*:\s*"([^"]+)"'),
        ("confidence", r'"confidence"\s*:\s*"([^"]+)"'),
        ("justificacion", r'"justificacion"\s*:\s*"([^"]{0,500})"'),
    ]:
        m = re.search(pattern, candidate, re.DOTALL)
        if m:
            result[field] = m.group(1)

    # evidencia_textual — intentar parsear solo ese bloque
    ev_match = re.search(r'"evidencia_textual"\s*:\s*(\{[^}]+\})', candidate, re.DOTALL)
    if ev_match:
        try:
            result["evidencia_textual"] = json.loads(ev_match.group(1))
        except json.JSONDecodeError:
            pass

    if result:
        return result
    raise ValueError(f"No se pudo parsear la respuesta: {candidate[:200]}")


def extract_ev(result: dict, key: str) -> str:
    """Extrae campo de evidencia_textual. El léxico se guarda como JSON string
    para poder deserializarlo después con json.loads()."""
    ev = result.get("evidencia_textual", {})
    if not isinstance(ev, dict):
        return ""
    val = ev.get(key, "")
    if isinstance(val, list):
        return json.dumps(val, ensure_ascii=False)
    return str(val) if val else ""


# ─────────────────────────────────────────────────────────────
# CLIENTES LLM
# ─────────────────────────────────────────────────────────────

def call_claude(titular, abstract, noticia):
    """Claude Sonnet 4.6 — F1=0.462 (Kennedy et al. 2025)."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        r = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1000, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(titular, abstract, noticia)}]
        )
        res = parse_json_response(r.content[0].text)
        res["label"] = normalize_label(res.get("label"))
        return res
    except Exception as e:
        return {"error": str(e), "label": None}


def call_gpt(titular, abstract, noticia):
    """GPT-5.2 — F1=0.448 (Kennedy et al. 2025)."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        r = client.chat.completions.create(
            model="gpt-5.2-2025-12-11", max_completion_tokens=1000, temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(titular, abstract, noticia)}
            ]
        )
        res = parse_json_response(r.choices[0].message.content)
        res["label"] = normalize_label(res.get("label"))
        return res
    except Exception as e:
        return {"error": str(e), "label": None}


def call_llama_groq(titular, abstract, noticia):
    """Llama 3.1 8B via Groq — gratis, ~500 tok/s. F1=0.397 (Kennedy et al. 2025)."""
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model="llama-3.1-8b-instant", max_tokens=1000, temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(titular, abstract, noticia)}
                ]
            )
            res = parse_json_response(r.choices[0].message.content)
            res["label"] = normalize_label(res.get("label"))
            return res
        except Exception as e:
            err_str = str(e)
            wait_match = re.search(r'Please try again in (\d+(?:\.\d+)?)s', err_str)
            if wait_match and attempt < 3:
                time.sleep(float(wait_match.group(1)) + 2)
                continue
            return {"error": err_str, "label": None}
    return {"error": "Max retries exceeded", "label": None}