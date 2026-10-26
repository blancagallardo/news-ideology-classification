"""
Re-run Claude solo para artículos donde falló, sin tocar GPT ni Llama.

Uso:
    python3 -m ground_truth.rerun_claude

Se puede ejecutar mientras rerun_llama está corriendo.
"""

import sys
import time
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.llm_clients import call_claude, extract_ev
from ground_truth.voting import apply_voting
from ground_truth.data import OUTPUT_CSV

PARQUET_PATH = Path(OUTPUT_CSV).with_suffix('.parquet')
CHECKPOINT_N = 20
DELAY_SEC    = 2.0   # evitar rate limit (429)

CLAUDE_COLS = [
    "claude_label", "claude_confianza", "claude_justificacion",
    "claude_error", "claude_lexico", "claude_encuadre", "claude_omisiones",
]
VOTING_COLS = [
    "final_label", "confianza_voto", "accion",
    "razonamiento_voto", "acuerdo_3llms",
]


def safe_save(df: pd.DataFrame):
    """Escribe solo columnas de Claude + votación para no machacar el progreso de Llama."""
    current = pd.read_parquet(PARQUET_PATH)
    for col in CLAUDE_COLS + VOTING_COLS:
        if col in df.columns:
            current[col] = df[col]
    current.to_parquet(PARQUET_PATH, index=False)
    current.to_csv(OUTPUT_CSV, index=False)


def reapply_voting(df: pd.DataFrame) -> pd.DataFrame:
    votes = df.apply(
        lambda r: apply_voting(r["claude_label"], r["gpt_label"], r["llama_label"]),
        axis=1
    )
    df["final_label"]       = votes.apply(lambda v: v["final_label"])
    df["confianza_voto"]    = votes.apply(lambda v: v["confianza"])
    df["accion"]            = votes.apply(lambda v: v["accion"])
    df["razonamiento_voto"] = votes.apply(lambda v: v["razonamiento_voto"])
    df["acuerdo_3llms"]     = (
        (df["claude_label"] == df["gpt_label"]) &
        (df["gpt_label"]    == df["llama_label"]) &
        df["claude_label"].notna()
    )
    return df


def main():
    print("=" * 60)
    print("RE-RUN CLAUDE — solo artículos con error anterior")
    print("GPT y Llama no se modifican")
    print("=" * 60)

    df = pd.read_parquet(PARQUET_PATH)

    mask_pending = df["claude_error"].notna() | df["claude_label"].isna()
    pending_idx  = df[mask_pending].index.tolist()

    print(f"\nTotal artículos   : {len(df):,}")
    print(f"Ya con Claude OK  : {(~mask_pending).sum():,}")
    print(f"Pendientes Claude : {len(pending_idx):,}")
    print(f"Tiempo estimado   : ~{len(pending_idx) * DELAY_SEC / 60:.0f} min\n")

    procesados = 0

    with tqdm(total=len(pending_idx), desc="Re-procesando Claude") as pbar:
        for i, idx in enumerate(pending_idx):
            row = df.loc[idx]

            res = call_claude(
                titular  = str(row.get("titular", "")),
                abstract = "",
                noticia  = str(row.get("noticia", "")),
            )

            # Rate limit → esperar y reintentar una vez
            if res.get("error") and "429" in str(res.get("error", "")):
                time.sleep(30)
                res = call_claude(
                    titular  = str(row.get("titular", "")),
                    abstract = "",
                    noticia  = str(row.get("noticia", "")),
                )

            df.at[idx, "claude_label"]         = res.get("label")
            df.at[idx, "claude_confianza"]     = res.get("confidence")
            df.at[idx, "claude_justificacion"] = res.get("justificacion")
            df.at[idx, "claude_error"]         = res.get("error")
            df.at[idx, "claude_lexico"]        = extract_ev(res, "lexico")
            df.at[idx, "claude_encuadre"]      = extract_ev(res, "encuadre")
            df.at[idx, "claude_omisiones"]     = extract_ev(res, "omisiones")

            procesados += 1
            pbar.update(1)

            if procesados % CHECKPOINT_N == 0:
                df_check = reapply_voting(df.copy())
                safe_save(df_check)
                pbar.set_postfix({"checkpoint": procesados})

            if i + 1 < len(pending_idx):
                time.sleep(DELAY_SEC)

    df = reapply_voting(df)
    safe_save(df)

    ok  = df["claude_label"].notna().sum()
    err = df["claude_error"].notna().sum()
    print(f"\nClaude OK: {ok:,} | Errores restantes: {err:,}")
    print(f"Guardado en: {PARQUET_PATH}")


if __name__ == "__main__":
    main()