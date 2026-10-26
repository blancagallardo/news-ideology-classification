#!/usr/bin/env python3
"""
run.py — Punto de entrada del scraper

Uso:
  python run.py                          # todos los medios
  python run.py --medios eldiario        # solo elDiario
  python run.py --medios eldiario 20min  # varios
  python run.py --reanudar               # continuar desde checkpoint
  python run.py --merge                  # solo mergear JSONs existentes
  python run.py --medios okdiario --temas politica economia  # temas concretos
  python run.py --años 3                 # historial de N años (default: 3)
"""

import argparse
import logging
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import config
import utils

from medios import eldiario, okdiario, elconfidencial, publico, elpais, elmundo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

MEDIOS = {
    "eldiario":       (eldiario, "elDiario.es"),
    "okdiario":       (okdiario, "OKDiario"),
    "elconfidencial": (elconfidencial, "El Confidencial"),
    "publico":        (publico, "Público"),
    "elpais":         (elpais, "El País"),
    "elmundo":        (elmundo, "El Mundo"),
}

ORDEN_RECOMENDADO = ["eldiario", "publico", "elconfidencial", "okdiario"]


def merge_all():
    """Une todos los JSONs de output/ en uno solo (TODOS.json)."""
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    todos = []
    seen = set()

    for f in Path(config.OUTPUT_DIR).glob("*.json"):
        if f.name == "TODOS.json":
            continue
        with open(f, "r", encoding="utf-8") as fp:
            articulos = json.load(fp)
        for art in articulos:
            url = art.get("url", "")
            if url not in seen:
                seen.add(url)
                todos.append(art)
        log.info(f"  {f.name}: {len(articulos)} artículos")

    todos.sort(key=lambda x: x.get("fecha", ""), reverse=True)
    out_path = f"{config.OUTPUT_DIR}/TODOS.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    log.info(f"\nMerge completo: {len(todos)} artículos únicos → {out_path}")

    # Estadísticas por medio y tema
    stats = {}
    for art in todos:
        medio = art.get("medio", "?")
        tema  = art.get("tema", "?")
        stats.setdefault(medio, {}).setdefault(tema, 0)
        stats[medio][tema] += 1

    log.info("\n── Estadísticas ──────────────────────────────")
    for medio, temas in sorted(stats.items()):
        total = sum(temas.values())
        detalle = "  |  ".join(f"{t}: {n}" for t, n in sorted(temas.items()))
        log.info(f"  {medio:<20} {total:>6} total   ({detalle})")
    log.info("──────────────────────────────────────────────")

    return out_path


def run_medio(nombre_medio, temas, session, reanudar, limite=None):
    modulo, nombre_display = MEDIOS[nombre_medio]
    log.info(f"\n{'='*55}")
    log.info(f"  INICIANDO: {nombre_display}")
    log.info(f"  Temas: {temas}")
    log.info(f"  Fecha límite: {config.FECHA_LIMITE.strftime('%Y-%m-%d')}")
    log.info(f"{'='*55}")

    checkpoint = utils.load_checkpoint(modulo.NOMBRE) if reanudar else {"urls": [], "articulos": []}
    t0 = time.time()

    try:
        articulos = modulo.scrape(session, temas, checkpoint, reanudar=reanudar, limite=limite)
    except KeyboardInterrupt:
        log.warning(f"\n[{nombre_display}] Interrupción detectada — el checkpoint ya está guardado.")
        raise

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    if nombre_medio != "eldiario":
        utils.save_output(modulo.NOMBRE, articulos)

    out_path = f"{config.OUTPUT_DIR}/{modulo.NOMBRE}.json"
    log.info(f"  {nombre_display}: {len(articulos)} artículos → {out_path}  [{mins}m {secs}s]")
    return articulos


def main():
    parser = argparse.ArgumentParser(description="Scraper de noticias españolas (3 años)")
    parser.add_argument("--medios", nargs="+", default=ORDEN_RECOMENDADO,
                        choices=list(MEDIOS.keys()) + ["todos"],
                        help="Medios a scrapear (default: todos)")
    parser.add_argument("--temas", nargs="+", default=config.TEMAS,
                        choices=["politica", "economia", "opinion"],
                        help="Temas a recopilar (default: todos)")
    parser.add_argument("--reanudar", action="store_true", default=True,
                        help="Continuar desde checkpoint guardado (default: True)")
    parser.add_argument("--reiniciar", action="store_true",
                        help="Ignorar checkpoint y empezar desde cero")
    parser.add_argument("--merge", action="store_true",
                        help="Solo unir los JSONs existentes, sin scrapear")
    parser.add_argument("--años", type=int, default=3,
                        help="Años de historial (default: 3)")
    parser.add_argument("--limite", type=int, default=None,
                        help="Máximo de artículos a scrapear por medio (para pruebas)")
    args = parser.parse_args()

    config.FECHA_LIMITE = datetime.now() - timedelta(days=365 * args.años)

    if args.reiniciar:
        args.reanudar = False

    if args.merge:
        merge_all()
        return

    medios_sel = ORDEN_RECOMENDADO if "todos" in args.medios else args.medios
    session = requests.Session()

    log.info("=" * 55)
    log.info("  SCRAPER DE NOTICIAS ESPAÑOLAS")
    log.info(f"  Medios : {', '.join(medios_sel)}")
    log.info(f"  Temas  : {', '.join(args.temas)}")
    log.info(f"  Rango  : últimos {args.años} años (desde {config.FECHA_LIMITE.strftime('%Y-%m-%d')})")
    log.info(f"  Reanudar: {'sí' if args.reanudar else 'no'}")
    log.info("=" * 55)

    resumen = {}
    t_total = time.time()

    try:
        for medio in medios_sel:
            articulos = run_medio(medio, args.temas, session, args.reanudar, limite=args.limite)
            resumen[medio] = len(articulos)
    except KeyboardInterrupt:
        log.warning("\nScraping interrumpido por el usuario.")

    log.info("\n── Resumen final ─────────────────────────────")
    for medio, n in resumen.items():
        _, nombre_display = MEDIOS[medio]
        log.info(f"  {nombre_display:<20} {n:>6} artículos")
    mins, secs = divmod(int(time.time() - t_total), 60)
    log.info(f"  Tiempo total: {mins}m {secs}s")
    log.info("──────────────────────────────────────────────")

    log.info("\nMergeando todos los resultados...")
    merge_all()


if __name__ == "__main__":
    main()