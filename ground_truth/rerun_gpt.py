"""
Re-run GPT solo para artículos donde falló, sin tocar Claude ni Llama.

Uso:
    python3 -m ground_truth.rerun_gpt

Se puede ejecutar mientras rerun_llama está corriendo.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.llm_clients import call_gpt, extract_ev
from ground_truth.voting import apply_voting
from ground_truth.data import OUTPUT_CSV

PARQUET_PATH = Path(OUTPUT_CSV).with_suffix('.parquet')
CHECKPOINT_N = 10

GPT_COLS = [
    "gpt_label", "gpt_confianza", "gpt_justificacion",
    "gpt_error", "gpt_lexico", "gpt_encuadre", "gpt_omisiones",
]
VOTING_COLS = [
    "final_label", "confianza_voto", "accion",
    "razonamiento_voto", "acuerdo_3llms",
]


def safe_save(df: pd.DataFrame):
    """Escribe solo columnas de GPT + votación para no machacar el progreso de Llama."""
    current = pd.read_parquet(PARQUET_PATH)
    for col in GPT_COLS + VOTING_COLS:
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
    print("RE-RUN GPT — solo artículos con error anterior")
    print("Claude y Llama no se modifican")
    print("=" * 60)

    df = pd.read_parquet(PARQUET_PATH)

    mask_pending = df["gpt_error"].notna() | df["gpt_label"].isna()
    pending_idx  = df[mask_pending].index.tolist()

    print(f"\nTotal artículos : {len(df):,}")
    print(f"Ya con GPT OK   : {(~mask_pending).sum():,}")
    print(f"Pendientes GPT  : {len(pending_idx):,}\n")

    procesados = 0

    with tqdm(total=len(pending_idx), desc="Re-procesando GPT") as pbar:
        for i, idx in enumerate(pending_idx):
            row = df.loc[idx]

            res = call_gpt(
                titular  = str(row.get("titular", "")),
                abstract = "",
                noticia  = str(row.get("noticia", "")),
            )

            df.at[idx, "gpt_label"]         = res.get("label")
            df.at[idx, "gpt_confianza"]     = res.get("confidence")
            df.at[idx, "gpt_justificacion"] = res.get("justificacion")
            df.at[idx, "gpt_error"]         = res.get("error")
            df.at[idx, "gpt_lexico"]        = extract_ev(res, "lexico")
            df.at[idx, "gpt_encuadre"]      = extract_ev(res, "encuadre")
            df.at[idx, "gpt_omisiones"]     = extract_ev(res, "omisiones")

            procesados += 1
            pbar.update(1)

            if procesados % CHECKPOINT_N == 0:
                df_check = reapply_voting(df.copy())
                safe_save(df_check)

    df = reapply_voting(df)
    safe_save(df)

    ok  = df["gpt_label"].notna().sum()
    err = df["gpt_error"].notna().sum()
    print(f"\nGPT OK: {ok:,} | Errores restantes: {err:,}")
    print(f"Guardado en: {PARQUET_PATH}")


if __name__ == "__main__":
    main()