# medios/elconfidencial.py — Scraper para ElConfidencial.com
# Usa la Wayback Machine CDX API para obtener URLs históricas (3 años)
# y luego scrappea el sitio real para el contenido.

import logging
import re
import requests
from bs4 import BeautifulSoup
from scrapers import utils
import config

log = logging.getLogger(__name__)

NOMBRE = "elconfidencial"

CDX_API = "http://web.archive.org/cdx/search/cdx"

# Secciones del dominio principal mapeadas a cada tema
SECCIONES_TEMA = {
    "politica": ["elconfidencial.com/espana/"],
    "economia":  ["elconfidencial.com/economia/", "elconfidencial.com/empresas/"],
    "opinion":   [],  # viene de blogs.elconfidencial.com
}

# Secciones de blogs (opinión) — todas las del subdominio de España y economía
BLOGS_TEMA = {
    "opinion": ["blogs.elconfidencial.com/espana/", "blogs.elconfidencial.com/economia/",
                "blogs.elconfidencial.com/mercados/", "blogs.elconfidencial.com/"],
}

# Patrón de artículo: termina en /YYYY-MM-DD/slug_ID/ o /slug_ID/
PATRON_ARTICULO_MAIN = re.compile(r"/\d{4}-\d{2}-\d{2}/[^/]+_\d+/?$")
PATRON_ARTICULO_BLOG = re.compile(r"/\d{4}-\d{2}-\d{2}/[^/]+_\d+/?$")

# Patrón de fecha en URL (para pre-filtrar por año)
PATRON_FECHA_URL = re.compile(r"/(20\d{2})-(\d{2})-(\d{2})/")

BODY_SELECTORS = [
    "div.innerArticle__body",
    "div.news-body-complete",
    "div.innerArticle__content",
    "div.newsType__content",
    "div.innerArticle",
    "article",
]


def _cdx_get_urls(url_pattern, fecha_desde, fecha_hasta, limit=100000):
    """Consulta la CDX API y devuelve lista de URLs únicas."""
    try:
        r = requests.get(CDX_API, params={
            "url":      url_pattern,
            "output":   "text",
            "fl":       "original",
            "filter":   ["statuscode:200", "mimetype:text/html"],
            "from":     fecha_desde.strftime("%Y%m%d"),
            "to":       fecha_hasta.strftime("%Y%m%d"),
            "collapse": "original",
            "limit":    limit,
        }, timeout=60)
        return [l for l in r.text.strip().split("\n") if l and "elconfidencial.com" in l]
    except Exception as e:
        log.error(f"[ElConfidencial] CDX error para {url_pattern}: {e}")
        return []


def get_urls(session, tema):
    """Obtiene URLs del tema usando CDX API + filtro por fecha en URL."""
    from datetime import datetime
    fecha_hasta = datetime.now()
    fecha_desde = config.FECHA_LIMITE

    all_urls = []
    seen = set()

    # Dominio principal (política y economía)
    for patron_cdx in SECCIONES_TEMA.get(tema, []):
        log.info(f"[ElConfidencial/{tema}] CDX: {patron_cdx}*")
        raw = _cdx_get_urls(patron_cdx + "*", fecha_desde, fecha_hasta)
        log.info(f"[ElConfidencial/{tema}] CDX devolvió {len(raw)} URLs brutas")

        for url in raw:
            if url in seen:
                continue
            if not PATRON_ARTICULO_MAIN.search(url):
                continue
            # Filtrar por fecha en URL
            m = PATRON_FECHA_URL.search(url)
            if m:
                fecha = utils.parse_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                if fecha and utils.date_is_too_old(fecha):
                    continue
            seen.add(url)
            fecha_hint = utils.parse_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}") if m else None
            all_urls.append((url, fecha_hint))

    # Blogs (opinión)
    if tema == "opinion":
        log.info(f"[ElConfidencial/opinion] CDX: blogs.elconfidencial.com/*")
        raw_blogs = _cdx_get_urls("blogs.elconfidencial.com/*", fecha_desde, fecha_hasta)
        log.info(f"[ElConfidencial/opinion] CDX blogs devolvió {len(raw_blogs)} URLs brutas")

        for url in raw_blogs:
            if url in seen:
                continue
            if not PATRON_ARTICULO_BLOG.search(url):
                continue
            m = PATRON_FECHA_URL.search(url)
            if m:
                fecha = utils.parse_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                if fecha and utils.date_is_too_old(fecha):
                    continue
            seen.add(url)
            fecha_hint = utils.parse_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}") if m else None
            all_urls.append((url, fecha_hint))

    log.info(f"[ElConfidencial/{tema}] Total URLs válidas: {len(all_urls)}")
    return all_urls


def parse_article(url, session, fecha_hint=None):
    r = utils.fetch(url, session)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "lxml")
    jld = utils.extract_json_ld(soup)

    # Titular
    titular = utils.clean_text(jld.get("headline", ""))
    if not titular:
        h1 = soup.find("h1")
        titular = utils.clean_text(h1.get_text()) if h1 else ""
    if not titular:
        og = soup.find("meta", {"property": "og:title"})
        if og:
            titular = utils.clean_text(og.get("content", ""))
    if not titular:
        return None

    # Abstract
    abstract = utils.clean_text(jld.get("description", ""))
    if not abstract:
        meta = soup.find("meta", {"property": "og:description"}) or \
               soup.find("meta", {"name": "description"})
        if meta:
            abstract = utils.clean_text(meta.get("content", ""))

    # Autor
    autor = ""
    autor_jld = jld.get("author", "")
    if isinstance(autor_jld, list):
        autor = ", ".join(a.get("name", "") for a in autor_jld if isinstance(a, dict))
    elif isinstance(autor_jld, dict):
        autor = autor_jld.get("name", "")

    # Fecha
    fecha = utils.parse_date(jld.get("datePublished", "")) or fecha_hint
    if not fecha:
        meta_fecha = soup.find("meta", {"property": "article:published_time"})
        if meta_fecha:
            fecha = utils.parse_date(meta_fecha.get("content", ""))
    if not fecha:
        m = PATRON_FECHA_URL.search(url)
        if m:
            fecha = utils.parse_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    if fecha and not utils.is_within_range(fecha):
        return None

    # Cuerpo
    noticia = utils.clean_text(jld.get("articleBody", ""))
    if not noticia or len(noticia) < 200:
        noticia = utils.extract_body(soup, BODY_SELECTORS)
    noticia = re.sub(r'(Te puede interesar|Lee también|Relacionado)[^\n]*', '', noticia)
    noticia = utils.clean_text(noticia)
    if not noticia:
        noticia = abstract

    tema = utils.clasificar_tema(titular, abstract, noticia)

    return {
        "medio":    "El Confidencial",
        "url":      url,
        "titular":  titular,
        "autor":    autor or "Redacción El Confidencial",
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
        log.info(f"[ElConfidencial] Tema: {tema}")
        urls = get_urls(session, tema)
        urls = [(u, f) for u, f in urls if u not in urls_vistas]
        log.info(f"[ElConfidencial/{tema}] {len(urls)} URLs nuevas")

        for url, fecha_hint in urls:
            if url in urls_vistas:
                continue
            try:
                art = parse_article(url, session, fecha_hint)
                if art:
                    articulos.append(art)
                    nuevos += 1
            except Exception as e:
                log.error(f"[ElConfidencial] Error {url}: {e}")
            finally:
                urls_vistas.add(url)
                checkpoint["urls"].append(url)

            if nuevos % 50 == 0 and nuevos > 0:
                checkpoint["articulos"] = articulos
                utils.save_checkpoint(NOMBRE, checkpoint)

            if limite and nuevos >= limite:
                log.info(f"[ElConfidencial] Límite de {limite} artículos alcanzado.")
                break

    checkpoint["articulos"] = articulos
    utils.save_checkpoint(NOMBRE, checkpoint)
    utils.save_output(NOMBRE, articulos)
    log.info(f"[ElConfidencial] Total nuevos: {nuevos}")
    return articulos
