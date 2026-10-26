"""
Re-run Llama (via Cerebras) para artículos pendientes tras Ollama.

Uso:
    python3 -m ground_truth.rerun_cerebras

- Para automáticamente al llegar a 950K tokens (margen de seguridad sobre 1M/día).
- Los artículos restantes los procesa rerun_groq.py.
- Claude y GPT NO se modifican.
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
from config import CEREBRAS_API_KEY

PARQUET_PATH   = Path(OUTPUT_CSV).with_suffix('.parquet')
CHECKPOINT_N   = 10
CONCURRENCY    = 8          # Cerebras es muy rápido
TOKEN_LIMIT    = 950_000    # margen sobre 1M/día
MODEL          = "llama3.1-8b"

LLAMA_COLS = [
    "llama_label", "llama_confianza", "llama_justificacion",
    "llama_error", "llama_lexico", "llama_encuadre", "llama_omisiones",
]
VOTING_COLS = [
    "final_label", "confianza_voto", "accion",
    "razonamiento_voto", "acuerdo_3llms",
]

_client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_API_KEY)


def call_cerebras_single(titular: str, noticia: str) -> dict:
    for attempt in range(4):
        try:
            r = _client.chat.completions.create(
                model=MODEL, max_tokens=600, temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(titular, "", noticia)},
                ]
            )
            tokens = r.usage.total_tokens if r.usage else 0
            res = parse_json_response(r.choices[0].message.content)
            res["label"]   = normalize_label(res.get("label"))
            res["_tokens"] = tokens
            return res
        except Exception as e:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            return {"error": str(e), "label": None, "_tokens": 0}
    return {"error": "Max retries exceeded", "label": None, "_tokens": 0}


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
    df.to_parquet(PARQUET_PATH, index=False)
    df.to_csv(OUTPUT_CSV, index=False)


async def main():
    if not CEREBRAS_API_KEY:
        print("ERROR: CEREBRAS_API_KEY no está en el .env")
        return

    print("=" * 60)
    print("RE-RUN LLAMA — Cerebras API (llama3.1-8b, ~2000 tok/s)")
    print(f"Límite de tokens: {TOKEN_LIMIT:,} / 1,000,000 diarios")
    print("Claude y GPT NO se modifican")
    print("=" * 60)

    df = pd.read_parquet(PARQUET_PATH)

    assert df["claude_label"].notna().sum() > 0, "claude_label vacío — revisar parquet"
    assert df["gpt_label"].notna().sum() > 0,    "gpt_label vacío — revisar parquet"

    mask_pending = df["llama_error"].notna() | df["llama_label"].isna()
    pending_idx  = df[mask_pending].index.tolist()
    n_pending    = len(pending_idx)

    tokens_est   = n_pending * 1_300
    arts_posible = min(n_pending, TOKEN_LIMIT // 1_300)

    print(f"\nTotal artículos  : {len(df):,}")
    print(f"Claude OK        : {df['claude_label'].notna().sum():,}")
    print(f"GPT OK           : {df['gpt_label'].notna().sum():,}")
    print(f"Llama OK         : {(~mask_pending).sum():,}")
    print(f"Pendientes       : {n_pending:,}")
    print(f"Tokens estimados : ~{tokens_est:,}  (límite: {TOKEN_LIMIT:,})")
    print(f"Procesará        : ~{arts_posible:,} artículos antes de parar")
    print(f"Tiempo estimado  : ~{arts_posible * 1_300 / 2_000 / 60:.0f} min")
    print()

    if n_pending == 0:
        print("Nada que procesar.")
        return

    loop           = asyncio.get_running_loop()
    sem            = asyncio.Semaphore(CONCURRENCY)
    procesados     = 0
    tokens_usados  = 0
    parado_limite  = False

    async def one(idx):
        async with sem:
            row = df.loc[idx]
            res = await loop.run_in_executor(
                None, call_cerebras_single,
                str(row.get("titular", "")),
                str(row.get("noticia", "")),
            )
            return idx, res

    tasks = [asyncio.create_task(one(idx)) for idx in pending_idx]

    with tqdm(total=n_pending, desc="Cerebras") as pbar:
        for coro in asyncio.as_completed(tasks):
            if tokens_usados >= TOKEN_LIMIT:
                # Cancelar tareas restantes
                for t in tasks:
                    t.cancel()
                parado_limite = True
                break

            try:
                idx, res = await coro
                tokens_usados += res.pop("_tokens", 0)
                apply_result(df, idx, res)
            except Exception as e:
                tqdm.write(f"Error: {e}")

            procesados += 1
            pbar.update(1)
            pbar.set_postfix({"tokens": f"{tokens_usados:,}", "guardado": procesados})

            if procesados % CHECKPOINT_N == 0:
                df_check = reapply_voting(df.copy())
                save(df_check)

    df = reapply_voting(df)
    save(df)

    print("\n" + "=" * 60)
    if parado_limite:
        restantes = df["llama_label"].isna().sum() + df["llama_error"].notna().sum()
        print(f"LÍMITE DE TOKENS ALCANZADO — lanza rerun_groq.py para los {restantes:,} restantes")
    else:
        print("COMPLETADO — todos los artículos procesados")
    print("=" * 60)
    print(f"Artículos procesados : {procesados:,}")
    print(f"Tokens usados        : {tokens_usados:,} / {TOKEN_LIMIT:,}")
    print(f"Claude intacto       : {df['claude_label'].notna().sum():,} / {len(df):,}")
    print(f"GPT intacto          : {df['gpt_label'].notna().sum():,} / {len(df):,}")
    print(f"\nGuardado en: {PARQUET_PATH}")


if __name__ == "__main__":
    asyncio.run(main())