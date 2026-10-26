# medios/publico.py — Scraper para Publico.es
# Estrategia: Sitemaps mensuales XML (mismo enfoque que elDiario)
# Sitemap index → sitemaps mensuales → URLs de artículos → scrape directo

import gzip
import logging
import re
from datetime import datetime
from bs4 import BeautifulSoup
from scrapers import utils
import config

log = logging.getLogger(__name__)

NOMBRE = "publico"

SITEMAP_INDEX = "https://www.publico.es/sitemap-index.xml"

# Sitemaps mensuales: sitemap-noticias-YYYYMM.xml o .xml.gz
SITEMAP_PATTERN = re.compile(r"sitemap-noticias-(\d{4})(\d{2})\.xml(\.gz)?")

SECCIONES_TEMA = {
    "politica": ["/politica/", "/espana/"],
    "economia":  ["/economia/"],
    "opinion":   ["/opinion/columnas/", "/opinion/tribunas/", "/opinion/vinetas/", "/opinion/"],
}

# Artículo: /seccion/slug.html  (sin fecha en URL, slug mínimo 4 chars)
PATRON_ARTICULO = re.compile(r"/[^/]+/[^/]{4,}\.html$")
EXCLUIR = re.compile(r"/(autor|autores|tag|tags|video|videos|amp|feed|ultima-edicion)/", re.IGNORECASE)

BODY_SELECTORS = [
    "div.article-body",
    "div#body-modules",
    "div.body-modules",
    "div[id='body-modules']",
    "main article",
    "article",
]


def _get_monthly_sitemaps(session):
    """Descarga el índice de sitemaps y devuelve los mensuales dentro del rango temporal."""
    r = utils.fetch(SITEMAP_INDEX, session)
    if not r:
        log.error("[Público] No se pudo descargar el sitemap index")
        return []

    soup = BeautifulSoup(r.content, "xml")
    sitemaps = []

    for loc_tag in soup.find_all("loc"):
        url = loc_tag.text.strip()
        m = SITEMAP_PATTERN.search(url)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        sitemap_date = datetime(year, month, 1)
        if sitemap_date >= config.FECHA_LIMITE.replace(day=1):
            sitemaps.append((url, year, month))

    sitemaps.sort(key=lambda x: (x[1], x[2]), reverse=True)
    log.info(f"[Público] Sitemaps mensuales en rango: {len(sitemaps)}")
    return sitemaps


def _fetch_sitemap_xml(url, session):
    """Descarga un sitemap .xml o .xml.gz y devuelve el contenido descomprimido."""
    r = utils.fetch(url, session)
    if not r:
        return None
    content = r.content
    if url.endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except Exception as e:
            log.warning(f"[Público] Error descomprimiendo {url}: {e}")
            return None
    return content


def get_urls(session, tema):
    """Extrae URLs del tema desde los sitemaps mensuales."""
    secciones = SECCIONES_TEMA.get(tema, [])
    urls = []
    seen = set()

    sitemaps = _get_monthly_sitemaps(session)

    for sitemap_url, year, month in sitemaps:
        log.info(f"[Público/{tema}] Sitemap {year}-{month:02d}...")
        content = _fetch_sitemap_xml(sitemap_url, session)
        if not content:
            continue

        soup = BeautifulSoup(content, "xml")
        found = 0

        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            lastmod = url_tag.find("lastmod")
            if not loc:
                continue

            article_url = loc.text.strip()

            if not any(sec in article_url for sec in secciones):
                continue
            if EXCLUIR.search(article_url):
                continue
            if not PATRON_ARTICULO.search(article_url):
                continue
            if article_url in seen:
                continue

            fecha = utils.parse_date(lastmod.text.strip()) if lastmod else None
            if fecha and utils.date_is_too_old(fecha):
                continue
            if fecha and not utils.is_within_range(fecha):
                continue

            seen.add(article_url)
            urls.append((article_url, fecha))
            found += 1

        log.info(f"[Público/{tema}] {year}-{month:02d}: {found} URLs")

    log.info(f"[Público/{tema}] Total URLs: {len(urls)}")
    return urls


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
        autor = ", ".join(
            a.get("name", "") if isinstance(a, dict) else str(a)
            for a in autor_jld
        )
    elif isinstance(autor_jld, dict):
        autor = autor_jld.get("name", "")
    elif isinstance(autor_jld, str):
        autor = autor_jld
    if not autor:
        tag = soup.select_one("a[rel='author'], span.author-name, a.autor")
        if tag:
            autor = utils.clean_text(tag.get_text())

    # Fecha
    fecha = utils.parse_date(jld.get("datePublished", "")) or fecha_hint
    if not fecha:
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
    noticia = utils.clean_text(jld.get("articleBody", ""))
    if not noticia or len(noticia) < 200:
        noticia = utils.extract_body(soup, BODY_SELECTORS)
    noticia = re.sub(r'(Te puede interesar|Lee también|Relacionado)[^\n]*', '', noticia)
    noticia = utils.clean_text(noticia)
    if not noticia:
        noticia = abstract

    tema = utils.clasificar_tema(titular, abstract, noticia)

    return {
        "medio":    "Público",
        "url":      url,
        "titular":  titular,
        "autor":    autor or "Redacción Público",
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
        log.info(f"[Público] Tema: {tema}")
        urls = get_urls(session, tema)
        urls = [(u, f) for u, f in urls if u not in urls_vistas]
        log.info(f"[Público/{tema}] {len(urls)} URLs nuevas")

        for url, fecha_hint in urls:
            if url in urls_vistas:
                continue
            try:
                art = parse_article(url, session, fecha_hint)
                if art:
                    articulos.append(art)
                    nuevos += 1
            except Exception as e:
                log.error(f"[Público] Error {url}: {e}")
            finally:
                urls_vistas.add(url)
                checkpoint["urls"].append(url)

            if nuevos % 50 == 0 and nuevos > 0:
                checkpoint["articulos"] = articulos
                utils.save_checkpoint(NOMBRE, checkpoint)
                log.info(f"[Público] {nuevos} artículos scrapeados")

            if limite and nuevos >= limite:
                log.info(f"[Público] Límite de {limite} artículos alcanzado.")
                break

    checkpoint["articulos"] = articulos
    utils.save_checkpoint(NOMBRE, checkpoint)
    utils.save_output(NOMBRE, articulos)
    log.info(f"[Público] Total nuevos: {nuevos}")
    return articulos
