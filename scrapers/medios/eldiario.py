import logging
import re
from bs4 import BeautifulSoup
from datetime import datetime
from scrapers import utils
import config

log = logging.getLogger(__name__)

NOMBRE = "eldiario"

SITEMAP_INDEX = "https://www.eldiario.es/sitemap_index_25b87.xml"
SITEMAP_PATTERN = re.compile(r"sitemap_contents_(\d{4})_(\d{2})_25b87_\d+\.xml")

# Qué secciones mapear a cada tema
SECCIONES_TEMA = {
    "politica": ["/politica/", "/espana/"],
    "economia":  ["/economia/"],
    "opinion":   ["/opinion/"],
}


def get_monthly_sitemaps(session):
    """Descarga el índice de sitemaps y devuelve los mensuales dentro del rango temporal."""
    r = utils.fetch(SITEMAP_INDEX, session)
    if not r:
        log.error("No se pudo descargar el sitemap index")
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
    log.info(f"[elDiario] Sitemaps mensuales en rango: {len(sitemaps)}")
    return sitemaps


def get_urls_from_sitemap(session, tema):
    """Extrae todas las URLs del tema desde los sitemaps mensuales."""
    secciones = SECCIONES_TEMA.get(tema, [])
    urls = []
    seen = set()

    sitemaps = get_monthly_sitemaps(session)

    for sitemap_url, year, month in sitemaps:
        log.info(f"[elDiario/{tema}] Procesando sitemap {year}-{month:02d}...")
        r = utils.fetch(sitemap_url, session)
        if not r:
            continue

        soup = BeautifulSoup(r.content, "xml")
        found = 0

        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            lastmod = url_tag.find("lastmod")
            if not loc:
                continue

            article_url = loc.text.strip()

            # Filtrar por sección del tema
            if not any(sec in article_url for sec in secciones):
                continue
            # Excluir páginas de autor/tag
            if any(x in article_url for x in ["/autores/", "/usuarios/", "/tags/"]):
                continue
            # Solo artículos con patrón _N_XXXXXXX.html (noticias: _1_, opinión: _129_, etc.)
            if not re.search(r"_\d+_\d+\.html$", article_url):
                continue
            if article_url in seen:
                continue

            # Filtrar por fecha
            fecha = utils.parse_date(lastmod.text.strip()) if lastmod else None
            if fecha and utils.date_is_too_old(fecha):
                continue
            if fecha and not utils.is_within_range(fecha):
                continue

            seen.add(article_url)
            urls.append((article_url, fecha))
            found += 1

        log.info(f"[elDiario/{tema}] {year}-{month:02d}: {found} URLs")

    log.info(f"[elDiario/{tema}] Total URLs: {len(urls)}")
    return urls


def parse_article(url, session):
    r = utils.fetch(url, session)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "lxml")
    jld = utils.extract_json_ld(soup)

    titular = utils.clean_text(jld.get("headline", ""))
    if not titular:
        h1 = soup.find("h1")
        titular = utils.clean_text(h1.get_text()) if h1 else ""
    if not titular:
        return None

    abstract = utils.clean_text(jld.get("description", ""))
    if not abstract:
        meta = soup.find("meta", {"name": "description"})
        if meta:
            abstract = utils.clean_text(meta.get("content", ""))

    # Autor
    autor_data = jld.get("author", "Redacción")
    if isinstance(autor_data, list):
        autor = ", ".join(a.get("name", "") for a in autor_data if isinstance(a, dict))
    elif isinstance(autor_data, dict):
        autor = autor_data.get("name", "Redacción")
    else:
        autor = str(autor_data)

    # Fecha
    fecha = utils.parse_date(jld.get("datePublished", ""))
    if not fecha:
        time_tag = soup.find("time")
        if time_tag:
            fecha = utils.parse_date(time_tag.get("datetime", ""))

    if fecha and not utils.is_within_range(fecha):
        return None

    # Cuerpo: JSON-LD articleBody es lo más limpio; fallback a div.c-content
    cuerpo = jld.get("articleBody", "")
    if not cuerpo or len(cuerpo) < 200:
        cuerpo = utils.extract_body(soup, ["div.c-content", "article.main-content", ".main-content"])

    return {
        "medio":    "elDiario.es",
        "url":      url,
        "titular":  utils.clean_text(titular),
        "autor":    utils.clean_text(autor),
        "fecha":    fecha or "",
        "abstract": abstract,
        "noticia":  utils.clean_text(cuerpo),
        "tema":     utils.clasificar_tema(titular, abstract, cuerpo),
    }


def scrape(session, temas, checkpoint, reanudar=False):
    articulos = checkpoint.get("articulos", []) if reanudar else []
    urls_procesadas = set(checkpoint.get("urls", [])) if reanudar else set()

    for tema in temas:
        log.info(f"[elDiario] --- TEMA: {tema} ---")
        lista_urls = get_urls_from_sitemap(session, tema)

        for url, fecha_hint in lista_urls:
            if url in urls_procesadas:
                continue
            try:
                art = parse_article(url, session)
                if art:
                    articulos.append(art)
            except Exception as e:
                log.error(f"[elDiario] Error {url}: {e}")
            finally:
                urls_procesadas.add(url)

            if len(articulos) % 50 == 0 and articulos:
                utils.save_checkpoint(NOMBRE, {
                    "urls": list(urls_procesadas),
                    "articulos": articulos,
                })

    utils.save_checkpoint(NOMBRE, {"urls": list(urls_procesadas), "articulos": articulos})
    utils.save_output(NOMBRE, articulos)
    log.info(f"[elDiario] Total: {len(articulos)} artículos")
    return articulos
