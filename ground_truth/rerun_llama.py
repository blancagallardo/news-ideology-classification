"""
Re-run Llama solo para artículos donde falló, sin tocar Claude ni GPT.

Uso:
    # Terminal 1 — arrancar Ollama con paralelismo (hazlo ANTES de ejecutar este script):
    OLLAMA_NUM_PARALLEL=4 ollama serve

    # Terminal 2:
    python3 -m ground_truth.rerun_llama

Ajusta CONCURRENCY según tu RAM:
    16 GB unified → CONCURRENCY = 4
    32 GB unified → CONCURRENCY = 6
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.llm_clients import build_prompt, parse_json_response, SYSTEM_PROMPT
from ground_truth.voting import apply_voting, normalize_label
from ground_truth.data import OUTPUT_CSV

PARQUET_PATH  = Path(OUTPUT_CSV).with_suffix('.parquet')
CHECKPOINT_N  = 10       # guardar cada N artículos procesados
CONCURRENCY   = 4        # requests paralelas a Ollama (ajusta según RAM)
MODEL         = "llama3.1:8b"

LLAMA_COLS = [
    "llama_label", "llama_confianza", "llama_justificacion",
    "llama_error", "llama_lexico", "llama_encuadre", "llama_omisiones",
]
VOTING_COLS = [
    "final_label", "confianza_voto", "accion",
    "razonamiento_voto", "acuerdo_3llms",
]

_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def call_llama_single(titular: str, noticia: str) -> dict:
    for attempt in range(4):
        try:
            r = _client.chat.completions.create(
                model=MODEL, max_tokens=600, temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(titular, "", noticia)},
                ]
            )
            res = parse_json_response(r.choices[0].message.content)
            res["label"] = normalize_label(res.get("label"))
            return res
        except Exception as e:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            return {"error": str(e), "label": None}
    return {"error": "Max retries exceeded", "label": None}


def reapply_voting(df: pd.DataFrame) -> pd.DataFrame:
    votes = df.apply(
        lambda r: apply_voting(r["claude_label"], r["gpt_label"], r["llama_label"]),
        axis=1,
    )
    v = pd.DataFrame(votes.tolist(), index=df.index)
    df["final_label"]       = v["final_label"]
    df["confianza_voto"]    = v["confianza"]
    df["accion"]            = v["accion"]
    df["razonamiento_voto"] = v["razonamiento_voto"]
    df["acuerdo_3llms"] = (
        (df["claude_label"] == df["gpt_label"]) &
        (df["gpt_label"]    == df["llama_label"]) &
        df["claude_label"].notna()
    )
    return df


def apply_result(df: pd.DataFrame, idx: int, res: dict):
    """Escribe solo columnas de Llama — nunca toca Claude ni GPT."""
    df.at[idx, "llama_label"]         = res.get("label")
    df.at[idx, "llama_confianza"]     = res.get("confidence")
    df.at[idx, "llama_justificacion"] = res.get("justificacion")
    df.at[idx, "llama_error"]         = res.get("error")

    ev = res.get("evidencia_textual", {})
    if isinstance(ev, dict):
        lexico = ev.get("lexico", "")
        df.at[idx, "llama_lexico"]    = json.dumps(lexico, ensure_ascii=False) if isinstance(lexico, list) else str(lexico)
        df.at[idx, "llama_encuadre"]  = str(ev.get("encuadre", ""))
        df.at[idx, "llama_omisiones"] = str(ev.get("omisiones", ""))


def save(df: pd.DataFrame):
    """Guarda parquet + CSV. df ya contiene todas las columnas (Claude, GPT, Llama)."""
    df.to_parquet(PARQUET_PATH, index=False)
    df.to_csv(OUTPUT_CSV, index=False)


async def process_batch(idxs: list, df: pd.DataFrame, sem: asyncio.Semaphore) -> list[tuple]:
    loop = asyncio.get_running_loop()

    async def one(idx):
        async with sem:
            row = df.loc[idx]
            res = await loop.run_in_executor(
                None, call_llama_single,
                str(row.get("titular", "")),
                str(row.get("noticia", "")),
            )
            return idx, res

    return await asyncio.gather(*[one(i) for i in idxs], return_exceptions=True)


async def main():
    print("=" * 60)
    print("RE-RUN LLAMA — Ollama local (llama3.1:8b)")
    print(f"Concurrencia: {CONCURRENCY} requests paralelas")
    print("Claude y GPT NO se modifican")
    print("=" * 60)

    df = pd.read_parquet(PARQUET_PATH)

    # Verificar integridad de Claude y GPT antes de tocar nada
    assert df["claude_label"].notna().sum() > 0, "claude_label está vacío — revisar parquet"
    assert df["gpt_label"].notna().sum() > 0,    "gpt_label está vacío — revisar parquet"

    mask_pending = df["llama_error"].notna() | df["llama_label"].isna()
    pending_idx  = df[mask_pending].index.tolist()
    n_pending    = len(pending_idx)

    secs_est = n_pending * 11 / CONCURRENCY
    print(f"\nTotal artículos  : {len(df):,}")
    print(f"Claude OK        : {df['claude_label'].notna().sum():,}")
    print(f"GPT OK           : {df['gpt_label'].notna().sum():,}")
    print(f"Llama OK         : {(~mask_pending).sum():,}")
    print(f"Pendientes       : {n_pending:,}")
    print(f"Tiempo estimado  : ~{secs_est/3600:.1f}h  ({secs_est/60:.0f} min)")
    print()

    if n_pending == 0:
        print("Nada que procesar.")
        return

    sem        = asyncio.Semaphore(CONCURRENCY)
    procesados = 0

    with tqdm(total=n_pending, desc=f"Ollama ×{CONCURRENCY}") as pbar:
        for i in range(0, n_pending, CHECKPOINT_N):
            batch_ids     = pending_idx[i:i + CHECKPOINT_N]
            batch_results = await process_batch(batch_ids, df, sem)

            for item in batch_results:
                if isinstance(item, Exception):
                    tqdm.write(f"Error batch: {item}")
                    continue
                idx, res = item
                apply_result(df, idx, res)
                procesados += 1

            pbar.update(len(batch_ids))
            df_check = reapply_voting(df.copy())
            save(df_check)
            pbar.set_postfix({"guardado": procesados})

    # Guardado final definitivo
    df = reapply_voting(df)
    save(df)

    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)
    print(f"Artículos procesados: {procesados:,}")
    print(f"Claude intacto      : {df['claude_label'].notna().sum():,} / {len(df):,}")
    print(f"GPT intacto         : {df['gpt_label'].notna().sum():,} / {len(df):,}")

    print(f"\nDistribución por acción:")
    for a, n in df["accion"].value_counts().items():
        print(f"  {a:<12}: {n:>5,}  ({n/len(df)*100:.1f}%)")

    incl = df[df["accion"] == "INCLUIR"]
    if len(incl):
        print(f"\nEtiquetas INCLUIR (n={len(incl):,}):")
        for lbl, n in incl["final_label"].value_counts().items():
            print(f"  {lbl:<12}: {n:>5,}  ({n/len(incl)*100:.1f}%)")

    print(f"\nGuardado en: {PARQUET_PATH}")


if __name__ == "__main__":
    asyncio.run(main())