"""
Medios y scores MBFC (escala -10 a +10):
    Público          -4.7  Left-Center
    ElDiario.es      -4.6  Left-Center
    El País          -2.5  Left-Center
    El Confidencial  +2.8  Right-Center  (estimado, sin página MBFC propia)
    El Mundo         +2.8  Right-Center
    OKDiario         +8.0  Right

Nota sobre limpieza:
    El corpus ya viene limpio de 2_eda.ipynb:
    - Noticias vacías o < 100 caracteres → eliminadas
    - Artículos truncados por paywall    → eliminados
    - Duplicados (medio+fecha+titular)   → eliminados
    - Liveblogs (en directo, última hora)→ eliminados
    Aquí solo se asignan scores MBFC y se descartan medios desconocidos.
"""

import json
import os
import unicodedata
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]  # TFM/ground_truth/data.py → TFM/ground_truth → TFM/

JSON_FILES = [
    ROOT / "data/processed/elpais_limpio_filtrado.json",
    ROOT / "data/processed/elmundo_limpio_filtrado.json",
    ROOT / "data/processed/okdiario_limpio_filtrado.json",
    ROOT / "data/processed/eldiario_limpio_filtrado.json",
    ROOT / "data/processed/publico_limpio_filtrado.json",
    ROOT / "data/processed/elconfidencial_limpio_filtrado.json",
]
# Nombre correcto del parquet generado por 2_eda.ipynb
PARQUET_FILE = str(ROOT / "data/processed/corpus_limpio_filtrado.parquet")

GT_DIR = ROOT / "data/ground_truth"
GT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_CSV = str(GT_DIR / "articulos_3000.csv")
OUTPUT_CSV = str(GT_DIR / "ground_truth_results.csv")

N_TOTAL = 3000

# ─────────────────────────────────────────────────────────────
# SCORES MBFC
# Escala: -10 (extrema izquierda) → 0 (centro) → +10 (extrema derecha)
# Fuente: mediabiasfactcheck.com
# El Confidencial: sin página MBFC propia, estimado como +2.8 (Right-Center)
#
# Una sola entrada por medio — _normalizar_medio() elimina espacios,
# acentos y puntos antes del lookup para hacer el match robusto.
# ─────────────────────────────────────────────────────────────

MEDIOS_MBFC = {
    "publico": {"score": -4.7, "label": "Left-Center"},
    "eldiario": {"score": -4.6, "label": "Left-Center"},
    "eldiarioes": {"score": -4.6, "label": "Left-Center"},  # elDiario.es con TLD
    "elpais": {"score": -2.5, "label": "Left-Center"},
    "elconfidencial": {"score": 2.8, "label": "Right-Center"},
    "elmundo": {"score": 2.8, "label": "Right-Center"},
    "okdiario": {"score": 8.0, "label": "Right"},
}

# Categorías originales MBFC — sin colapsar
MBFC_CLASES = ["Left-Center", "Right-Center", "Right"]


def _normalizar_medio(medio: str) -> str:
    """
    Normaliza el nombre del medio para el lookup en MEDIOS_MBFC.
    Elimina mayúsculas, espacios, puntos, guiones y acentos.
    Ejemplos:
        "El País"     → "elpais"
        "elDiario.es" → "eldiario"
        "Público"     → "publico"
        "OK Diario"   → "okdiario"
    """
    medio = medio.lower().strip()
    medio = unicodedata.normalize("NFD", medio)
    medio = "".join(c for c in medio if unicodedata.category(c) != "Mn")
    medio = medio.replace(" ", "").replace(".", "").replace("-", "")
    return medio


def get_mbfc_info(medio: str) -> dict:
    if not isinstance(medio, str):
        return {"score": None, "label": "unknown"}
    return MEDIOS_MBFC.get(_normalizar_medio(medio), {"score": None, "label": "unknown"})


# ─────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────

def cargar_corpus() -> pd.DataFrame:
    """
    Carga el corpus desde parquet (prioridad) o desde los JSONs individuales.
    Asigna automáticamente mbfc_score y mbfc_label a cada artículo.
    """
    if os.path.exists(PARQUET_FILE):
        print(f"Cargando parquet: {PARQUET_FILE}")
        df = pd.read_parquet(PARQUET_FILE)
        print(f"  {len(df):,} artículos")
    else:
        dfs = []
        for fpath in JSON_FILES:
            if not fpath.exists():
                print(f"  AVISO: no encontrado {fpath}")
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            df_medio = pd.DataFrame(data)
            print(f"  {fpath.name}: {len(df_medio):,} artículos")
            dfs.append(df_medio)
        if not dfs:
            raise FileNotFoundError(
                "No se encontró el parquet ni los JSONs en data/processed/.\n"
                "Ejecuta primero el notebook 2_eda.ipynb para generar el corpus."
            )
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Total: {len(df):,} artículos de {len(dfs)} medios")

    # Asignar scores MBFC
    mbfc_info = df["medio"].apply(get_mbfc_info)
    df["mbfc_score"] = mbfc_info.apply(lambda x: x["score"])
    df["mbfc_label"] = mbfc_info.apply(lambda x: x["label"])

    n_unknown = (df["mbfc_label"] == "unknown").sum()
    if n_unknown > 0:
        print(f"  AVISO: {n_unknown:,} artículos con medio no reconocido en MBFC")

    print("\nDistribución por medio:")
    for medio, n in df["medio"].value_counts().items():
        score = df[df["medio"] == medio]["mbfc_score"].iloc[0]
        label = df[df["medio"] == medio]["mbfc_label"].iloc[0]
        score_str = f"{score:+.1f}" if score is not None else "N/A"
        print(f"  {medio:20s}  {label:15s} ({score_str})  {n:,} artículos")

    return df


# ─────────────────────────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────────────────────────

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    El corpus ya viene limpio de 2_eda.ipynb (noticias vacías,
    truncadas, duplicadas y liveblogs eliminados).
    Solo descartamos artículos de medios no reconocidos en MBFC.
    """
    n_antes = len(df)
    df = df[df["mbfc_label"] != "unknown"].reset_index(drop=True)
    n_eliminados = n_antes - len(df)
    if n_eliminados > 0:
        print(f"  Medios desconocidos eliminados: {n_eliminados:,}")
    print(f"  Dataset listo: {len(df):,} artículos")
    return df


# ─────────────────────────────────────────────────────────────
# MUESTREO ESTRATIFICADO
# ─────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n_total: int = N_TOTAL) -> pd.DataFrame:
    """
    Muestra estratificada con la MISMA cantidad por medio.
    6 medios × 2.500 = 15.000 artículos.
    """
    medios_disponibles = sorted(df["medio"].unique())
    n_medios = len(medios_disponibles)
    n_per_medio = n_total // n_medios

    print(f"\nMuestreo: {n_per_medio:,} artículos × {n_medios} medios = {n_medios * n_per_medio:,} total")
    samples = []

    for medio in medios_disponibles:
        subset = df[df["medio"] == medio]
        mbfc_s = subset["mbfc_score"].iloc[0]
        mbfc_l = subset["mbfc_label"].iloc[0]
        score_str = f"{mbfc_s:+.1f}" if mbfc_s is not None else "N/A"

        if len(subset) >= n_per_medio:
            sampled = subset.sample(n_per_medio, random_state=42)
            print(f"  {medio:20s} ({score_str} {mbfc_l:15s}): {n_per_medio:,} de {len(subset):,}")
        else:
            sampled = subset
            print(f"  {medio:20s} ({score_str} {mbfc_l:15s}): AVISO — solo {len(subset):,} disponibles")

        samples.append(sampled)

    result = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=42)
    print(f"\nTotal muestra: {len(result):,} artículos")
    return result