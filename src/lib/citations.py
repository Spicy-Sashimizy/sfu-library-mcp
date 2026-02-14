"""Citation extraction and formatting for PNX records.

Extracted from the monolith. Supports APA, MLA, Chicago, BibTeX, and RIS formats.
Implements DATA-005: citation format validation.
"""

import logging
import re

from lib.validators import normalize_encoding

logger = logging.getLogger("sfu_library_mcp")


def extract_metadata(item: dict) -> dict | None:
    """Extract metadata from a PNX record for citation generation."""
    if not item:
        return None

    pnx = item.get("pnx", {})
    display = pnx.get("display", {})
    addata = pnx.get("addata", {})
    control = pnx.get("control", {})

    metadata = {
        "title": normalize_encoding(display.get("title", [""])[0]),
        "creators": display.get("creator", []),
        "contributors": display.get("contributor", []),
        "date": display.get("creationdate", [""])[0],
        "publisher": normalize_encoding(display.get("publisher", [""])[0]),
        "type": display.get("type", [""])[0].lower(),
        "source": normalize_encoding(display.get("source", [""])[0]),
        "isbn": addata.get("isbn", [""])[0] if addata.get("isbn") else "",
        "issn": addata.get("issn", [""])[0] if addata.get("issn") else "",
        "doi": addata.get("doi", [""])[0] if addata.get("doi") else "",
        "volume": addata.get("volume", [""])[0] if addata.get("volume") else "",
        "issue": addata.get("issue", [""])[0] if addata.get("issue") else "",
        "spage": addata.get("spage", [""])[0] if addata.get("spage") else "",
        "epage": addata.get("epage", [""])[0] if addata.get("epage") else "",
        "pages": addata.get("pages", [""])[0] if addata.get("pages") else "",
        "record_id": control.get("recordid", [""])[0] if control.get("recordid") else "",
    }

    authors = metadata["creators"] if metadata["creators"] else metadata["contributors"]
    metadata["authors"] = authors

    doc_type = metadata["type"]
    if "article" in doc_type or "journal" in doc_type:
        metadata["resource_type"] = "article"
    elif "book" in doc_type:
        metadata["resource_type"] = "book"
    else:
        metadata["resource_type"] = "other"

    return metadata


def format_author_apa(author_string: str) -> str:
    """Format a single author for APA style (Last, F. M.)."""
    if not author_string:
        return ""
    author = author_string.split("$$")[0].strip()

    if "," in author:
        parts = author.split(",", 1)
        last = parts[0].strip()
        first = parts[1].strip() if len(parts) > 1 else ""
        initials = " ".join([n[0] + "." for n in first.split() if n])
        return f"{last}, {initials}" if initials else last
    else:
        parts = author.split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = " ".join([n[0] + "." for n in parts[:-1] if n])
            return f"{last}, {initials}"
        return author


def format_author_mla(author_string: str) -> str:
    """Format a single author for MLA style (Last, First Middle)."""
    if not author_string:
        return ""
    return author_string.split("$$")[0].strip()


def format_apa_citation(metadata: dict | None) -> str:
    """Generate APA 7th edition citation."""
    if not metadata:
        return "Unable to generate citation: no metadata available."

    parts = []

    authors = metadata.get("authors", [])
    if authors:
        if len(authors) == 1:
            parts.append(format_author_apa(authors[0]))
        elif len(authors) == 2:
            parts.append(f"{format_author_apa(authors[0])} & {format_author_apa(authors[1])}")
        elif len(authors) <= 20:
            author_list = ", ".join([format_author_apa(a) for a in authors[:-1]])
            parts.append(f"{author_list}, & {format_author_apa(authors[-1])}")
        else:
            author_list = ", ".join([format_author_apa(a) for a in authors[:19]])
            parts.append(f"{author_list}, ... {format_author_apa(authors[-1])}")

    year = metadata.get("date", "n.d.")
    if year:
        year = year[:4] if len(year) >= 4 else year
    parts.append(f"({year}).")

    title = metadata.get("title", "Untitled")
    resource_type = metadata.get("resource_type", "other")

    if resource_type == "article":
        parts.append(f"{title}.")
        source = metadata.get("source", "")
        if source:
            journal_part = source
            vol = metadata.get("volume", "")
            issue = metadata.get("issue", "")
            if vol:
                journal_part += f", {vol}"
                if issue:
                    journal_part += f"({issue})"
            pages = metadata.get("pages", "") or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip("-")
            if pages:
                journal_part += f", {pages}"
            parts.append(f"{journal_part}.")
    else:
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")

    doi = metadata.get("doi", "")
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    return " ".join(parts)


def format_mla_citation(metadata: dict | None) -> str:
    """Generate MLA 9th edition citation."""
    if not metadata:
        return "Unable to generate citation: no metadata available."

    parts = []

    authors = metadata.get("authors", [])
    if authors:
        if len(authors) == 1:
            parts.append(format_author_mla(authors[0]) + ".")
        elif len(authors) == 2:
            parts.append(f"{format_author_mla(authors[0])}, and {format_author_mla(authors[1])}.")
        else:
            parts.append(f"{format_author_mla(authors[0])}, et al.")

    title = metadata.get("title", "Untitled")
    resource_type = metadata.get("resource_type", "other")

    if resource_type == "article":
        parts.append(f'"{title}."')
        source = metadata.get("source", "")
        if source:
            journal_part = source
            vol = metadata.get("volume", "")
            issue = metadata.get("issue", "")
            if vol:
                journal_part += f", vol. {vol}"
            if issue:
                journal_part += f", no. {issue}"
            year = metadata.get("date", "")[:4] if metadata.get("date") else ""
            if year:
                journal_part += f", {year}"
            pages = metadata.get("pages", "") or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip("-")
            if pages:
                journal_part += f", pp. {pages}"
            parts.append(journal_part + ".")
    else:
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        year = metadata.get("date", "")[:4] if metadata.get("date") else ""
        if publisher and year:
            parts.append(f"{publisher}, {year}.")
        elif publisher:
            parts.append(f"{publisher}.")
        elif year:
            parts.append(f"{year}.")

    doi = metadata.get("doi", "")
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    return " ".join(parts)


def format_chicago_citation(metadata: dict | None) -> str:
    """Generate Chicago 17th edition citation (notes-bibliography style)."""
    if not metadata:
        return "Unable to generate citation: no metadata available."

    parts = []

    authors = metadata.get("authors", [])
    if authors:
        if len(authors) == 1:
            parts.append(format_author_mla(authors[0]) + ".")
        elif len(authors) <= 3:
            author_list = ", ".join([format_author_mla(a) for a in authors[:-1]])
            parts.append(f"{author_list}, and {format_author_mla(authors[-1])}.")
        else:
            parts.append(f"{format_author_mla(authors[0])}, et al.")

    title = metadata.get("title", "Untitled")
    resource_type = metadata.get("resource_type", "other")

    if resource_type == "article":
        parts.append(f'"{title}."')
        source = metadata.get("source", "")
        if source:
            journal_part = source
            vol = metadata.get("volume", "")
            issue = metadata.get("issue", "")
            if vol:
                journal_part += f" {vol}"
            if issue:
                journal_part += f", no. {issue}"
            year = metadata.get("date", "")[:4] if metadata.get("date") else ""
            if year:
                journal_part += f" ({year})"
            pages = metadata.get("pages", "") or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip("-")
            if pages:
                journal_part += f": {pages}"
            parts.append(journal_part + ".")
    else:
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        year = metadata.get("date", "")[:4] if metadata.get("date") else ""
        if publisher:
            parts.append(f"{publisher}, {year}." if year else f"{publisher}.")

    doi = metadata.get("doi", "")
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    return " ".join(parts)


def format_bibtex_entry(metadata: dict | None) -> str:
    """Generate BibTeX entry."""
    if not metadata:
        return "% Unable to generate citation: no metadata available."

    resource_type = metadata.get("resource_type", "other")

    authors = metadata.get("authors", [])
    first_author = authors[0].split("$$")[0].split(",")[0].strip() if authors else "unknown"
    first_author = "".join(c for c in first_author if c.isalnum())
    year = metadata.get("date", "")[:4] if metadata.get("date") else "YYYY"
    title_word = metadata.get("title", "untitled").split()[0] if metadata.get("title") else "untitled"
    title_word = "".join(c for c in title_word if c.isalnum())
    key = f"{first_author.lower()}{year}{title_word.lower()}"

    entry_type = "article" if resource_type == "article" else "book"

    lines = [f"@{entry_type}{{{key},"]

    if authors:
        author_str = " and ".join([a.split("$$")[0].strip() for a in authors])
        lines.append(f"  author = {{{author_str}}},")

    title = metadata.get("title", "")
    if title:
        lines.append(f"  title = {{{title}}},")

    if year and year != "YYYY":
        lines.append(f"  year = {{{year}}},")

    if resource_type == "article":
        source = metadata.get("source", "")
        if source:
            lines.append(f"  journal = {{{source}}},")
        vol = metadata.get("volume", "")
        if vol:
            lines.append(f"  volume = {{{vol}}},")
        issue = metadata.get("issue", "")
        if issue:
            lines.append(f"  number = {{{issue}}},")
        pages = metadata.get("pages", "") or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip("-")
        if pages:
            lines.append(f"  pages = {{{pages}}},")
    else:
        publisher = metadata.get("publisher", "")
        if publisher:
            lines.append(f"  publisher = {{{publisher}}},")

    isbn = metadata.get("isbn", "")
    if isbn:
        lines.append(f"  isbn = {{{isbn}}},")

    issn = metadata.get("issn", "")
    if issn:
        lines.append(f"  issn = {{{issn}}},")

    doi = metadata.get("doi", "")
    if doi:
        lines.append(f"  doi = {{{doi}}},")

    lines.append("}")

    return "\n".join(lines)


def format_ris_entry(metadata: dict | None) -> str:
    """Generate RIS format entry (for EndNote, Zotero, etc.)."""
    if not metadata:
        return "TY  - GEN\nER  -"

    resource_type = metadata.get("resource_type", "other")

    lines = []

    if resource_type == "article":
        lines.append("TY  - JOUR")
    else:
        lines.append("TY  - BOOK")

    for author in metadata.get("authors", []):
        author_clean = author.split("$$")[0].strip()
        lines.append(f"AU  - {author_clean}")

    title = metadata.get("title", "")
    if title:
        lines.append(f"TI  - {title}")

    year = metadata.get("date", "")[:4] if metadata.get("date") else ""
    if year:
        lines.append(f"PY  - {year}")

    if resource_type == "article":
        source = metadata.get("source", "")
        if source:
            lines.append(f"JO  - {source}")
        vol = metadata.get("volume", "")
        if vol:
            lines.append(f"VL  - {vol}")
        issue = metadata.get("issue", "")
        if issue:
            lines.append(f"IS  - {issue}")
        spage = metadata.get("spage", "")
        if spage:
            lines.append(f"SP  - {spage}")
        epage = metadata.get("epage", "")
        if epage:
            lines.append(f"EP  - {epage}")
    else:
        publisher = metadata.get("publisher", "")
        if publisher:
            lines.append(f"PB  - {publisher}")

    isbn = metadata.get("isbn", "")
    if isbn:
        lines.append(f"SN  - {isbn}")

    issn = metadata.get("issn", "")
    if issn:
        lines.append(f"SN  - {issn}")

    doi = metadata.get("doi", "")
    if doi:
        lines.append(f"DO  - {doi}")

    lines.append("ER  -")

    return "\n".join(lines)


def extract_full_text_links(item: dict) -> dict | None:
    """Extract all full-text access links from a record."""
    if not item:
        return None

    pnx = item.get("pnx", {})
    links = pnx.get("links", {})
    addata = pnx.get("addata", {})

    result = {
        "html_links": [],
        "pdf_links": [],
        "source_links": [],
        "doi_url": None,
        "open_access": False,
    }

    for link in links.get("linktohtml", []):
        if isinstance(link, str):
            result["html_links"].append(link)

    for link in links.get("linktopdf", []):
        if isinstance(link, str):
            result["pdf_links"].append(link)

    for link in links.get("linktorsrc", []):
        if isinstance(link, str):
            result["source_links"].append(link)

    doi = addata.get("doi", [""])[0] if addata.get("doi") else ""
    if doi:
        if doi.startswith("http"):
            result["doi_url"] = doi
        else:
            result["doi_url"] = f"https://doi.org/{doi}"

    oa_indicators = links.get("openaccess", []) or links.get("openurl", [])
    result["open_access"] = len(oa_indicators) > 0

    return result
