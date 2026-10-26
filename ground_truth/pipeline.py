"""
Ground Truth Pipeline — Clasificación Ideológica de Noticias en Español

Uso:
    # Paso 1 — cargar, limpiar y muestrear (genera articulos_15000.csv)
    python -m ground_truth.pipeline sample

    # Paso 2 — clasificar con los 3 LLMs
    python -m ground_truth.pipeline
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .data import (
    cargar_corpus, limpiar_dataset, stratified_sample,
    get_mbfc_info, SAMPLE_CSV, OUTPUT_CSV, N_TOTAL,
)
from .llm_clients import call_claude, call_gpt, call_llama_groq, extract_ev
from .voting import apply_voting, IZQ, DER

BATCH_SIZE = 20
SLEEP_SEC  = 0.5

# Left-Center → IZQUIERDA, Right-Center / Right → DERECHA
_MBFC_A_ETIQUETA = {
    "Left-Center":  IZQ,
    "Right-Center": DER,
    "Right":        DER,
}


# ─────────────────────────────────────────────────────────────
# PROCESAR UN ARTÍCULO
# ─────────────────────────────────────────────────────────────

async def process_article(row: pd.Series) -> dict:
    titular    = str(row.get("titular", ""))
    abstract   = str(row.get("abstract", ""))
    noticia    = str(row.get("noticia", ""))
    medio      = str(row.get("medio", "Desconocido"))
    article_id = str(row.get("id", row.name))
    fecha      = str(row.get("fecha", ""))
    tema       = str(row.get("tema", ""))

    mbfc       = get_mbfc_info(medio)
    mbfc_score = mbfc["score"]
    mbfc_label = mbfc["label"]

    loop = asyncio.get_running_loop()
    claude_r, gpt_r, llama_r = await asyncio.gather(
        loop.run_in_executor(None, call_claude,     titular, abstract, noticia),
        loop.run_in_executor(None, call_gpt,        titular, abstract, noticia),
        loop.run_in_executor(None, call_llama_groq, titular, abstract, noticia),
    )

    vote = apply_voting(claude_r.get("label"), gpt_r.get("label"), llama_r.get("label"))

    c_label  = claude_r.get("label")
    g_label  = gpt_r.get("label")
    ll_label = llama_r.get("label")

    acuerdo_3llms = (c_label == g_label == ll_label) and c_label is not None
    diverge_llama = (
        c_label is not None and g_label is not None and ll_label is not None
        and c_label == g_label and ll_label != c_label
    )

    etiqueta_esperada  = _MBFC_A_ETIQUETA.get(mbfc_label)
    final_label        = vote["final_label"]
    coincide_con_medio = (final_label == etiqueta_esperada) if etiqueta_esperada and final_label else None

    return {
        # Metadatos
        "id":                    article_id,
        "medio":                 medio,
        "mbfc_score":            mbfc_score,
        "mbfc_label":            mbfc_label,
        "fecha":                 fecha,
        "tema":                  tema,
        "titular":               titular,
        "n_palabras_noticia":    len(noticia.split()),
        # Claude
        "claude_label":          c_label,
        "claude_confianza":      claude_r.get("confidence"),
        "claude_justificacion":  claude_r.get("justificacion"),
        "claude_lexico":         extract_ev(claude_r, "lexico"),
        "claude_encuadre":       extract_ev(claude_r, "encuadre"),
        "claude_omisiones":      extract_ev(claude_r, "omisiones"),
        "claude_error":          claude_r.get("error"),
        # GPT
        "gpt_label":             g_label,
        "gpt_confianza":         gpt_r.get("confidence"),
        "gpt_justificacion":     gpt_r.get("justificacion"),
        "gpt_lexico":            extract_ev(gpt_r, "lexico"),
        "gpt_encuadre":          extract_ev(gpt_r, "encuadre"),
        "gpt_omisiones":         extract_ev(gpt_r, "omisiones"),
        "gpt_error":             gpt_r.get("error"),
        # Llama
        "llama_label":           ll_label,
        "llama_confianza":       llama_r.get("confidence"),
        "llama_justificacion":   llama_r.get("justificacion"),
        "llama_lexico":          extract_ev(llama_r, "lexico"),
        "llama_encuadre":        extract_ev(llama_r, "encuadre"),
        "llama_omisiones":       extract_ev(llama_r, "omisiones"),
        "llama_error":           llama_r.get("error"),
        # Votación
        "final_label":           final_label,
        "confianza_voto":        vote["confianza"],
        "accion":                vote["accion"],
        "razonamiento_voto":     vote["razonamiento_voto"],
        # Métricas
        "acuerdo_3llms":         acuerdo_3llms,
        "diverge_llama":         diverge_llama,
        "coincide_con_medio":    coincide_con_medio,
        "etiqueta_esperada":     etiqueta_esperada,
    }


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

async def run_pipeline(n_test: int = 0):
    print("=" * 65)
    print("GROUND TRUTH PIPELINE — Noticias en Español")
    print("Claude Sonnet 4.6 + GPT-5.2 + Llama 3.1 8B (Groq)")
    print("=" * 65)

    df = pd.read_csv(SAMPLE_CSV)
    print(f"Artículos cargados: {len(df):,}")

    if n_test > 0:
        df = df.head(n_test)
        print(f"Modo test: procesando solo {n_test} artículos")

    output_path = Path(OUTPUT_CSV)
    if output_path.exists():
        done_df  = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["id"].astype(str))
        df       = df[~df["id"].astype(str).isin(done_ids)]
        print(f"Retomando: {len(df):,} pendientes ({len(done_ids):,} ya procesados)")
    else:
        done_df = pd.DataFrame()

    results = []

    with tqdm(total=len(df), desc="Clasificando") as pbar:
        for i in range(0, len(df), BATCH_SIZE):
            batch         = df.iloc[i:i + BATCH_SIZE]
            tasks         = [process_article(row) for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in batch_results:
                if isinstance(res, Exception):
                    print(f"\nError: {res}")
                else:
                    results.append(res)

            pbar.update(len(batch))

            if results and len(results) % 20 == 0:
                _save(done_df, results, OUTPUT_CSV)
                print(f"\nCheckpoint: {len(done_df) + len(results):,} procesados")

            if i + BATCH_SIZE < len(df):
                await asyncio.sleep(SLEEP_SEC)

    _save(done_df, results, OUTPUT_CSV)
    _print_stats(OUTPUT_CSV)


def _save(done_df, results, path):
    df_out = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
    df_out.to_csv(path, index=False)
    # También en parquet — preserva tipos y evita problemas con comillas/comas
    df_out.to_parquet(path.replace(".csv", ".parquet"), index=False)


def _print_stats(path):
    df = pd.read_csv(path)
    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print("=" * 65)
    print(f"Total procesados: {len(df):,}\n")

    print("Por acción:")
    for accion, n in df["accion"].value_counts().items():
        print(f"  {accion:12s}: {n:6,} ({n/len(df)*100:.1f}%)")

    incluidos = df[df["accion"] == "INCLUIR"]
    if len(incluidos) > 0:
        print(f"\nEtiquetas finales (INCLUIR, n={len(incluidos):,}):")
        for label, n in incluidos["final_label"].value_counts().items():
            print(f"  {label:12s}: {n:6,} ({n/len(incluidos)*100:.1f}%)")

        print("\nCoincidencia con ideología del medio (INCLUIR):")
        for medio, grp in incluidos.groupby("medio"):
            score     = grp["mbfc_score"].iloc[0]
            score_str = f"{score:+.1f}" if pd.notna(score) else "N/A"
            pct       = grp["coincide_con_medio"].mean() * 100 if "coincide_con_medio" in grp else float("nan")
            print(f"  {medio:20s} ({score_str}): {pct:.1f}% coinciden con su medio")

        print("\nAcuerdo 3/3 entre LLMs (INCLUIR):")
        if "acuerdo_3llms" in incluidos.columns:
            n_acuerdo = incluidos["acuerdo_3llms"].sum()
            print(f"  {n_acuerdo:,} de {len(incluidos):,} ({n_acuerdo/len(incluidos)*100:.1f}%)")

    print(f"\nGuardado en: {path}")
    print(f"Artículos para entrenar: {len(incluidos):,}")


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        print("=" * 65)
        print("PASO 1 — Carga, limpieza y muestreo estratificado")
        print("=" * 65)
        df      = cargar_corpus()
        df      = limpiar_dataset(df)
        muestra = stratified_sample(df, n_total=N_TOTAL)
        muestra.to_csv(SAMPLE_CSV, index=False)
        print(f"\nMuestra guardada: {SAMPLE_CSV} ({len(muestra):,} artículos)")
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        asyncio.run(run_pipeline(n_test=n))
    else:
        asyncio.run(run_pipeline())