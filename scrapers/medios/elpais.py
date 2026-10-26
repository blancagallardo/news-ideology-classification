# medios/elpais.py — Scraper para ElPais.com
# CDX API → URL archivada de Wayback Machine (evita paywall)

import logging
import re
import requests as req
from bs4 import BeautifulSoup
from scrapers import utils
import config

log = logging.getLogger(__name__)

NOMBRE = "elpais"
CDX_API = "http://web.archive.org/cdx/search/cdx"

SECCIONES_TEMA = {
    "politica": ["elpais.com/espana/", "elpais.com/politica/"],
    "economia":  ["elpais.com/economia/"],
    "opinion":   ["elpais.com/opinion/"],
}

# El País usa dos formatos de URL según la época:
# Nuevo: /espana/2023-03-28/titulo.html       (YYYY-MM-DD)
# Viejo: /politica/2019/01/15/titulo_N.html   (YYYY/MM/DD)
PATRON_ARTICULO = re.compile(
    r"(/20\d{2}-\d{2}-\d{2}/[a-z0-9_-]+\.html$"   # formato nuevo
    r"|/20\d{2}/\d{2}/\d{2}/[a-z0-9_]+\.html$)",   # formato viejo
    re.IGNORECASE
)
# Captura fecha tanto en formato YYYY-MM-DD como YYYY/MM/DD
PATRON_FECHA_URL = re.compile(r"/(20\d{2})[-/](\d{2})[-/](\d{2})/")
EXCLUIR = re.compile(r"/(autor|tag|tags|fotogaleria|video|podcast|amp|suscripcion)/", re.IGNORECASE)

BODY_SELECTORS = [
    "div.a_c",
    "div[data-dtm-region='articulo_cuerpo']",
    "div.article-body",
    "div.articulo-cuerpo",
    "article",
]

def _contenido_truncado(noticia: str, abstract: str) -> bool:
    """
    True si no se pudo obtener el cuerpo completo del artículo:
    - El cuerpo está vacío o es idéntico al abstract (se usó el fallback)
    - El cuerpo es extremadamente corto (< 200 chars)
    """
    if not noticia:
        return True
    if abstract and noticia.strip() == abstract.strip():
        return True
    if len(noticia) < 200:
        return True
    return False


def _cdx_get_url_timestamps(url_pattern, fecha_desde, fecha_hasta, limit=500000):
    """Devuelve lista de (timestamp, url_original) desde CDX API."""
    try:
        r = req.get(CDX_API, params={
            "url":      url_pattern,
            "output":   "text",
            "fl":       "timestamp,original",
            "filter":   ["statuscode:200", "mimetype:text/html"],
            "from":     fecha_desde.strftime("%Y%m%d"),
            "to":       fecha_hasta.strftime("%Y%m%d"),
            "collapse": "original",
            "limit":    limit,
        }, timeout=90)
        pares = []
        for line in r.text.strip().split("\n"):
            parts = line.strip().split(" ", 1)
            if len(parts) == 2 and "elpais.com" in parts[1]:
                pares.append((parts[0], parts[1]))
        return pares
    except Exception as e:
        log.error(f"[ElPaís] CDX error para {url_pattern}: {e}")
        return []


def get_urls(session, tema):
    """Obtiene lista de (url_canónica, url_wayback, fecha_hint)."""
    from datetime import datetime
    fecha_hasta = datetime.now()
    fecha_desde = config.FECHA_LIMITE

    all_urls = []
    seen = set()

    for patron_cdx in SECCIONES_TEMA.get(tema, []):
        log.info(f"[ElPaís/{tema}] CDX: {patron_cdx}*")
        pares = _cdx_get_url_timestamps(patron_cdx + "*", fecha_desde, fecha_hasta)
        log.info(f"[ElPaís/{tema}] CDX devolvió {len(pares)} entradas brutas")

        for timestamp, url in pares:
            url_clean = url.split("?")[0].split("#")[0]

            if url_clean in seen:
                continue
            if EXCLUIR.search(url_clean):
                continue
            if not PATRON_ARTICULO.search(url_clean):
                continue

            m = PATRON_FECHA_URL.search(url_clean)
            if m:
                fecha_hint = utils.parse_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                if fecha_hint and utils.date_is_too_old(fecha_hint):
                    continue
            else:
                fecha_hint = None

            seen.add(url_clean)
            wayback_url = f"https://web.archive.org/web/{timestamp}/{url_clean}"
            all_urls.append((url_clean, wayback_url, fecha_hint))

    log.info(f"[ElPaís/{tema}] Total URLs válidas: {len(all_urls)}")
    return all_urls


def parse_article(url_canonical, url_wayback, session, fecha_hint=None):
    r = utils.fetch(url_wayback, session)
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
        autor = ", ".join(
            a.get("name", "") if isinstance(a, dict) else str(a)
            for a in autor_jld
        )
    elif isinstance(autor_jld, dict):
        autor = autor_jld.get("name", "")
    elif isinstance(autor_jld, str):
        autor = autor_jld
    if not autor:
        tag = soup.select_one("a.a_md_a_n, a[rel='author'], span.author-name")
        if tag:
            autor = utils.clean_text(tag.get_text())

    # Fecha
    fecha = utils.parse_date(jld.get("datePublished", "")) or fecha_hint
    if not fecha:
        meta_fecha = soup.find("meta", {"property": "article:published_time"})
        if meta_fecha:
            fecha = utils.parse_date(meta_fecha.get("content", ""))
    if not fecha:
        m = PATRON_FECHA_URL.search(url_canonical)
        if m:
            fecha = utils.parse_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    if fecha and not utils.is_within_range(fecha):
        return None

    # Cuerpo
    noticia = utils.clean_text(jld.get("articleBody", ""))
    if not noticia or len(noticia) < 200:
        noticia = utils.extract_body(soup, BODY_SELECTORS)
    noticia = re.sub(r'(Te puede interesar|Lee también|Relacionado|Más información)[^\n]*', '', noticia)
    noticia = utils.clean_text(noticia)
    if not noticia:
        noticia = abstract

    truncado = _contenido_truncado(noticia, abstract)
    if truncado:
        log.debug(f"[ElPaís] Contenido truncado/incompleto: {url_canonical}")

    tema = utils.clasificar_tema(titular, abstract, noticia)

    return {
        "medio":    "El País",
        "url":      url_canonical,
        "titular":  titular,
        "autor":    autor or "Redacción El País",
        "fecha":    fecha or "",
        "abstract": abstract,
        "noticia":  noticia,
        "tema":     tema,
        "truncado": truncado,
    }


def scrape(session, temas, checkpoint, reanudar=True, limite=None):
    urls_vistas = set(checkpoint["urls"]) if reanudar else set()
    articulos = checkpoint["articulos"] if reanudar else []
    nuevos = 0
    truncados = 0

    for tema in temas:
        log.info(f"[ElPaís] Tema: {tema}")
        urls = get_urls(session, tema)
        urls = [(uc, uw, fh) for uc, uw, fh in urls if uc not in urls_vistas]
        log.info(f"[ElPaís/{tema}] {len(urls)} URLs nuevas")

        for url_canonical, url_wayback, fecha_hint in urls:
            if url_canonical in urls_vistas:
                continue
            try:
                art = parse_article(url_canonical, url_wayback, session, fecha_hint)
                if art:
                    articulos.append(art)
                    nuevos += 1
                    if art["truncado"]:
                        truncados += 1
            except Exception as e:
                log.error(f"[ElPaís] Error {url_canonical}: {e}")
            finally:
                urls_vistas.add(url_canonical)
                checkpoint["urls"].append(url_canonical)

            if nuevos % 50 == 0 and nuevos > 0:
                checkpoint["articulos"] = articulos
                utils.save_checkpoint(NOMBRE, checkpoint)
                log.info(f"[ElPaís] {nuevos} artículos ({truncados} truncados por paywall)")

            if limite and nuevos >= limite:
                log.info(f"[ElPaís] Límite de {limite} artículos alcanzado.")
                break

    checkpoint["articulos"] = articulos
    utils.save_checkpoint(NOMBRE, checkpoint)
    utils.save_output(NOMBRE, articulos)
    log.info(f"[ElPaís] Total nuevos: {nuevos} ({truncados} con paywall detectado)")
    return articulos
