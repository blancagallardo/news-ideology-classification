# medios/okdiario.py — Scraper para OKDiario.com
# Usa la Wayback Machine CDX API para obtener URLs históricas (3 años)

import logging
import re
import requests as req
from bs4 import BeautifulSoup
from scrapers import utils
import config

log = logging.getLogger(__name__)

NOMBRE = "okdiario"

CDX_API = "http://web.archive.org/cdx/search/cdx"

# Secciones de OKDiario que mapean a cada tema
SECCIONES_TEMA = {
    "politica": ["okdiario.com/espana/"],
    "economia":  ["okdiario.com/economia/"],
    "opinion":   ["okdiario.com/opinion/", "okdiario.com/editoriales/", "okdiario.com/tribunas/"],
}

# Patrones a excluir (no son artículos)
EXCLUIR = re.compile(r"/(autores?|author|tag|tags|page|archivos|category|feed|sitemap)/")
# Un artículo real de OKDiario termina en slug sin extensión, opcionalmente con /
PATRON_ARTICULO = re.compile(r"/[^/]{10,}/?$")

BODY_SELECTORS = [
    "div.entry-content",
    "div.post-body",
    "div.article-body",
    "article .content",
    "div.td-post-content",
]


def _cdx_get_urls(url_pattern, fecha_desde, fecha_hasta, limit=100000):
    """Consulta la CDX API y devuelve lista de URLs únicas."""
    try:
        r = req.get(CDX_API, params={
            "url":      url_pattern,
            "output":   "text",
            "fl":       "original",
            "filter":   ["statuscode:200", "mimetype:text/html"],
            "from":     fecha_desde.strftime("%Y%m%d"),
            "to":       fecha_hasta.strftime("%Y%m%d"),
            "collapse": "original",
            "limit":    limit,
        }, timeout=90)
        return [l for l in r.text.strip().split("\n") if l and "okdiario.com" in l]
    except Exception as e:
        log.error(f"[OKDiario] CDX error para {url_pattern}: {e}")
        return []


def get_urls(session, tema):
    """Obtiene URLs del tema usando la Wayback Machine CDX API."""
    from datetime import datetime
    fecha_hasta = datetime.now()
    fecha_desde = config.FECHA_LIMITE

    all_urls = []
    seen = set()

    for patron_cdx in SECCIONES_TEMA.get(tema, []):
        log.info(f"[OKDiario/{tema}] CDX: {patron_cdx}*")
        raw = _cdx_get_urls(patron_cdx + "*", fecha_desde, fecha_hasta)
        log.info(f"[OKDiario/{tema}] CDX devolvió {len(raw)} URLs brutas")

        for url in raw:
            if url in seen:
                continue
            if EXCLUIR.search(url):
                continue
            if "?" in url or "#" in url:
                continue
            if not PATRON_ARTICULO.search(url):
                continue
            seen.add(url)
            all_urls.append((url, None))  # sin fecha_hint (no hay fecha en URL)

    log.info(f"[OKDiario/{tema}] Total URLs válidas: {len(all_urls)}")
    return all_urls


def parse_article(url, session, fecha_hint=None):
    r = utils.fetch(url, session)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "lxml")

    # Titular
    titular = ""
    h1 = soup.find("h1")
    if h1:
        titular = utils.clean_text(h1.get_text())
    if not titular:
        og = soup.find("meta", {"property": "og:title"})
        if og:
            titular = utils.clean_text(og.get("content", ""))
    if not titular:
        return None

    # Abstract
    abstract = ""
    meta_desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
    if meta_desc:
        abstract = utils.clean_text(meta_desc.get("content", ""))

    # Autor
    autor = ""
    for sel in [".td-author-name a", ".author-name", "a[rel='author']", ".byline", "span.author"]:
        tag = soup.select_one(sel)
        if tag:
            autor = utils.clean_text(tag.get_text())
            break

    # Fecha
    fecha = fecha_hint
    meta_fecha = soup.find("meta", {"property": "article:published_time"})
    if meta_fecha:
        fecha = utils.parse_date(meta_fecha.get("content", ""))
    if not fecha:
        time_tag = soup.find("time")
        if time_tag:
            fecha = utils.parse_date(time_tag.get("datetime", ""))

    if fecha and not utils.is_within_range(fecha):
        return None

    # Cuerpo
    noticia = utils.extract_body(soup, BODY_SELECTORS)
    noticia = re.sub(r'(LEER MÁS|Lee también|Relacionado)[^\n]*', '', noticia)
    noticia = utils.clean_text(noticia)
    if not noticia:
        noticia = abstract

    tema = utils.clasificar_tema(titular, abstract, noticia)

    return {
        "medio":    "OKDiario",
        "url":      url,
        "titular":  titular,
        "autor":    autor or "Redacción OKDiario",
        "fecha":    fecha or "",
        "abstract": abstract,
        "noticia":  noticia,
        "tema":     tema,
    }


def scrape(session, temas, checkpoint, reanudar=True, limite=None):
    urls_vistas = set(checkpoint["urls"]) if reanudar else set()
    articulos = checkpoint["articulos"] if reanudar else []
    nuevos = 0

    for tema in temas:
        log.info(f"[OKDiario] Tema: {tema}")
        urls = get_urls(session, tema)
        urls = [(u, f) for u, f in urls if u not in urls_vistas]
        log.info(f"[OKDiario/{tema}] {len(urls)} URLs nuevas")

        for url, fecha_hint in urls:
            if url in urls_vistas:
                continue
            try:
                art = parse_article(url, session, fecha_hint)
                if art:
                    articulos.append(art)
                    nuevos += 1
            except Exception as e:
                log.error(f"[OKDiario] Error {url}: {e}")
            finally:
                urls_vistas.add(url)
                checkpoint["urls"].append(url)

            if nuevos % 50 == 0 and nuevos > 0:
                checkpoint["articulos"] = articulos
                utils.save_checkpoint(NOMBRE, checkpoint)

            if limite and nuevos >= limite:
                log.info(f"[OKDiario] Límite de {limite} artículos alcanzado.")
                break

    checkpoint["articulos"] = articulos
    utils.save_checkpoint(NOMBRE, checkpoint)
    log.info(f"[OKDiario] Total nuevos: {nuevos}")
    return articulos
