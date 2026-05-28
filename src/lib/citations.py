"""Citation extraction and formatting for PNX records.

Extracted from the monolith. Supports APA, MLA, Chicago, BibTeX, and RIS formats.
Implements DATA-005: citation format validation.
Strategies A/C/D for CDI record support.
"""

import logging
import re

from lib.validators import normalize_encoding

logger = logging.getLogger("sfu_library_mcp")


def enrich_metadata_from_crossref(metadata: dict) -> dict:
    """Fill in missing citation fields using CrossRef API.

    Only called when key fields (authors, date, source) are missing
    and a DOI is available. Returns the same metadata dict, enriched.
    """
    doi = metadata.get("doi", "")
    if not doi:
        return metadata

    # Only call CrossRef if critical fields are missing
    missing_authors = not metadata.get("authors")
    missing_date = not metadata.get("date")
    missing_source = not metadata.get("source")

    if not (missing_authors or missing_date or missing_source):
        return metadata

    try:
        from lib.openalex import fetch_crossref_work
        cr = fetch_crossref_work(doi)
    except Exception:
        return metadata

    if not cr:
        return metadata

    logger.info("Enriching metadata from CrossRef for DOI %s", doi)

    if missing_authors and cr.get("authors"):
        metadata["authors"] = cr["authors"]
        metadata["creators"] = cr["authors"]
    if missing_date and cr.get("date"):
        metadata["date"] = cr["date"]
    if missing_source and cr.get("source"):
        metadata["source"] = cr["source"]

    # Also fill in other missing fields
    for field in ("volume", "issue", "spage", "epage", "pages", "publisher"):
        if not metadata.get(field) and cr.get(field):
            metadata[field] = cr[field]

    return metadata


def _first(d: dict, k: str) -> str:
    """Return the first element of a PNX list field, or "" if missing/empty.

    PNX records may contain a present key mapped to an empty list, so the
    `dict.get(k, [""])[0]` idiom is unsafe (it raises IndexError on `[]`).
    """
    v = d.get(k) or [""]
    return v[0] if v else ""


def _is_cdi_record(control: dict) -> bool:
    """Check if a record is from CDI (external source) vs local ALMA."""
    record_id = control.get("recordid", [""])[0] if control.get("recordid") else ""
    return record_id.startswith("TN_cdi_")


def _extract_authors(display: dict, addata: dict, search: dict) -> list[str]:
    """Extract properly separated author list, handling CDI semicolon format.

    CDI records jam all authors into a single semicolon-delimited string
    in display.creator (e.g. "Smith, J ; Doe, A"). We detect this and
    fall back to addata.au or search.creator which have proper lists.
    """
    creators = display.get("creator", [])

    # Check for CDI semicolon-joined format: single entry containing " ; "
    if len(creators) == 1 and " ; " in creators[0]:
        # Prefer addata.au (properly separated individual authors)
        addata_authors = addata.get("au", [])
        if addata_authors:
            return addata_authors
        # Fall back to search.creator
        search_authors = search.get("creator", [])
        if search_authors and not (len(search_authors) == 1 and " ; " in search_authors[0]):
            return search_authors
        # Last resort: split the semicolon string ourselves
        return [a.strip() for a in creators[0].split(" ; ") if a.strip()]

    if creators:
        return creators

    # No display.creator at all — try addata.au, then search.creator, then contributors
    addata_authors = addata.get("au", [])
    if addata_authors:
        return addata_authors
    search_authors = search.get("creator", [])
    if search_authors:
        return search_authors
    return display.get("contributor", [])


def _extract_date(display: dict, addata: dict, search: dict) -> str:
    """Extract publication date with CDI fallbacks.

    CDI records often lack display.creationdate but have addata.date
    (full date like "2020-07-01") or search.creationdate (year "2020").
    """
    date = _first(display, "creationdate")
    if date:
        return date
    # addata.date has full ISO date — take just the year portion
    ad_date = addata.get("date", [""])[0] if addata.get("date") else ""
    if ad_date:
        return ad_date[:4]
    # search.creationdate has year
    s_date = search.get("creationdate", [""])[0] if search.get("creationdate") else ""
    return s_date


def _extract_journal_name(display: dict, addata: dict, is_cdi: bool) -> str:
    """Extract journal/source name, preferring addata.jtitle for CDI.

    CDI records set display.source to the database name (e.g. "Scopus")
    rather than the journal title. addata.jtitle has the actual journal.
    """
    # For CDI records, always prefer addata.jtitle
    if is_cdi:
        jtitle = addata.get("jtitle", [""])[0] if addata.get("jtitle") else ""
        if jtitle:
            return normalize_encoding(jtitle)
    # For ALMA records or fallback
    source = _first(display, "source")
    if source:
        return normalize_encoding(source)
    # Final fallback to addata.jtitle
    jtitle = addata.get("jtitle", [""])[0] if addata.get("jtitle") else ""
    return normalize_encoding(jtitle) if jtitle else ""


def extract_metadata(item: dict) -> dict | None:
    """Extract metadata from a PNX record for citation generation.

    Handles both ALMA (local) and CDI (external) records by using
    addata/search sections as fallbacks when display fields are missing
    or malformed (Strategy A).
    """
    if not item:
        return None

    pnx = item.get("pnx", {})
    display = pnx.get("display", {})
    addata = pnx.get("addata", {})
    control = pnx.get("control", {})
    search = pnx.get("search", {})

    is_cdi = _is_cdi_record(control)
    authors = _extract_authors(display, addata, search)
    date = _extract_date(display, addata, search)
    source = _extract_journal_name(display, addata, is_cdi)

    metadata = {
        "title": normalize_encoding(_first(display, "title")),
        "creators": authors,
        "contributors": display.get("contributor", []),
        "date": date,
        "publisher": normalize_encoding(_first(display, "publisher")),
        "type": _first(display, "type").lower(),
        "source": source,
        "isbn": addata.get("isbn", [""])[0] if addata.get("isbn") else "",
        "issn": addata.get("issn", [""])[0] if addata.get("issn") else "",
        "doi": addata.get("doi", [""])[0] if addata.get("doi") else "",
        "volume": addata.get("volume", [""])[0] if addata.get("volume") else "",
        "issue": addata.get("issue", [""])[0] if addata.get("issue") else "",
        "spage": addata.get("spage", [""])[0] if addata.get("spage") else "",
        "epage": addata.get("epage", [""])[0] if addata.get("epage") else "",
        "pages": addata.get("pages", [""])[0] if addata.get("pages") else "",
        "record_id": control.get("recordid", [""])[0] if control.get("recordid") else "",
        "is_cdi": is_cdi,
    }

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
    elif resource_type in ("magazine_article", "newspaper_article"):
        parts.append(f"{title}.")
        source = metadata.get("source", "")
        if source:
            pub_part = source
            pages = metadata.get("pages", "")
            if pages:
                pub_part += f", {pages}"
            parts.append(pub_part + ".")
    elif resource_type == "book":
        edition = metadata.get("edition", "")
        title_part = f"{title} ({edition} ed.)" if edition else title
        parts.append(f"{title_part}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type == "book_section":
        parts.append(f"{title}.")
        editors = metadata.get("editors", [])
        book_title = metadata.get("book_title", "")
        pages = metadata.get("pages", "")
        pages_part = f" (pp. {pages})" if pages else ""
        if editors and book_title:
            ed_str = ", ".join(editors)
            ed_label = "Eds." if len(editors) > 1 else "Ed."
            parts.append(f"In {ed_str} ({ed_label}), {book_title}{pages_part}.")
        elif book_title:
            parts.append(f"In {book_title}{pages_part}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type == "conference_paper":
        parts.append(f"{title} [Conference presentation].")
        conf = metadata.get("conference_name", "") or metadata.get("source", "")
        loc = metadata.get("conference_location", "")
        if conf:
            parts.append((f"{conf}, {loc}." if loc else f"{conf}."))
    elif resource_type == "thesis":
        degree = (metadata.get("degree_type", "") or "").lower()
        label = (
            "Master's thesis" if degree in ("masters", "master's", "master")
            else "Doctoral dissertation"
        )
        institution = metadata.get("institution", "")
        bracket = f"[{label}, {institution}]" if institution else f"[{label}]"
        parts.append(f"{title} {bracket}.")
    elif resource_type == "webpage":
        parts.append(f"{title}.")
        source = metadata.get("source", "")
        if source:
            parts.append(f"{source}.")
        accessed = metadata.get("accessed_date", "")
        if accessed:
            parts.append(f"Retrieved {accessed}.")
    elif resource_type == "software":
        version = metadata.get("version", "")
        ver_str = f" (Version {version})" if version else ""
        parts.append(f"{title}{ver_str} [Computer software].")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type == "report":
        report_number = metadata.get("report_number", "")
        rn = f" (Report No. {report_number})" if report_number else ""
        parts.append(f"{title}{rn}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type == "dataset":
        parts.append(f"{title} [Data set].")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type in ("film", "video"):
        label = "Film" if resource_type == "film" else "Video"
        parts.append(f"{title} [{label}].")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type == "audio":
        parts.append(f"{title} [Audio recording].")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type == "patent":
        parts.append(f"{title} [Patent].")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    elif resource_type == "presentation":
        parts.append(f"{title} [Presentation].")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")
    else:
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(publisher + ".")

    doi = metadata.get("doi", "")
    url = metadata.get("url", "")
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)
    elif url:
        parts.append(url)

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
    year = metadata.get("date", "")[:4] if metadata.get("date") else ""

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
            if year:
                journal_part += f", {year}"
            pages = metadata.get("pages", "") or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip("-")
            if pages:
                journal_part += f", pp. {pages}"
            parts.append(journal_part + ".")
    elif resource_type in ("magazine_article", "newspaper_article"):
        parts.append(f'"{title}."')
        source = metadata.get("source", "")
        if source:
            pub_part = source
            if year:
                pub_part += f", {year}"
            pages = metadata.get("pages", "")
            if pages:
                pub_part += f", p. {pages}"
            parts.append(pub_part + ".")
    elif resource_type == "book":
        parts.append(f"{title}.")
        edition = metadata.get("edition", "")
        publisher = metadata.get("publisher", "")
        edition_str = f"{edition} ed., " if edition else ""
        if publisher and year:
            parts.append(f"{edition_str}{publisher}, {year}.")
        elif publisher:
            parts.append(f"{edition_str}{publisher}.")
        elif year:
            parts.append(f"{edition_str}{year}.")
    elif resource_type == "book_section":
        parts.append(f'"{title}."')
        book_title = metadata.get("book_title", "")
        editors = metadata.get("editors", [])
        publisher = metadata.get("publisher", "")
        pages = metadata.get("pages", "")
        if book_title:
            ed_part = f"edited by {', '.join(editors)}, " if editors else ""
            section = f"{book_title}, {ed_part}"
            if publisher:
                section += f"{publisher}, "
            if year:
                section += f"{year}"
            if pages:
                section += f", pp. {pages}"
            parts.append(section + ".")
    elif resource_type == "conference_paper":
        parts.append(f'"{title}."')
        conf = metadata.get("conference_name", "") or metadata.get("source", "")
        if conf:
            parts.append((f"{conf}, {year}." if year else f"{conf}."))
    elif resource_type == "thesis":
        parts.append(f"{title}.")
        degree = (metadata.get("degree_type", "") or "").lower()
        degree_label = (
            "master's thesis" if degree in ("masters", "master's", "master")
            else "doctoral dissertation"
        )
        institution = metadata.get("institution", "")
        if institution and year:
            parts.append(f"{year}, {institution}, {degree_label}.")
        elif institution:
            parts.append(f"{institution}, {degree_label}.")
        elif year:
            parts.append(f"{year}, {degree_label}.")
    elif resource_type == "webpage":
        parts.append(f'"{title}."')
        source = metadata.get("source", "")
        if source:
            parts.append(f"{source},")
        if year:
            parts.append(f"{year}.")
    elif resource_type == "software":
        parts.append(f"{title}.")
        version = metadata.get("version", "")
        publisher = metadata.get("publisher", "")
        if version:
            parts.append(f"Version {version},")
        if publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year}.")
    elif resource_type == "report":
        parts.append(f"{title}.")
        report_number = metadata.get("report_number", "")
        publisher = metadata.get("publisher", "")
        if report_number:
            parts.append(f"Report No. {report_number},")
        if publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year}.")
    elif resource_type == "dataset":
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher and year:
            parts.append(f"{publisher}, {year}.")
        elif publisher:
            parts.append(f"{publisher}.")
    elif resource_type in ("film", "video"):
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher and year:
            parts.append(f"{publisher}, {year}.")
        elif publisher:
            parts.append(f"{publisher}.")
    elif resource_type == "audio":
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher and year:
            parts.append(f"{publisher}, {year}.")
        elif publisher:
            parts.append(f"{publisher}.")
    elif resource_type in ("patent", "presentation"):
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher and year:
            parts.append(f"{publisher}, {year}.")
    else:
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher and year:
            parts.append(f"{publisher}, {year}.")
        elif publisher:
            parts.append(f"{publisher}.")
        elif year:
            parts.append(f"{year}.")

    doi = metadata.get("doi", "")
    url = metadata.get("url", "")
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)
    elif url:
        parts.append(url)

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
    year = metadata.get("date", "")[:4] if metadata.get("date") else ""

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
            if year:
                journal_part += f" ({year})"
            pages = metadata.get("pages", "") or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip("-")
            if pages:
                journal_part += f": {pages}"
            parts.append(journal_part + ".")
    elif resource_type in ("magazine_article", "newspaper_article"):
        parts.append(f'"{title}."')
        source = metadata.get("source", "")
        if source:
            pub_part = source
            if year:
                pub_part += f", {year}"
            pages = metadata.get("pages", "")
            if pages:
                pub_part += f", {pages}"
            parts.append(pub_part + ".")
    elif resource_type == "book":
        parts.append(f"{title}.")
        edition = metadata.get("edition", "")
        publisher = metadata.get("publisher", "")
        edition_str = f"{edition} ed. " if edition else ""
        if publisher:
            parts.append(f"{edition_str}{publisher}, {year}." if year else f"{edition_str}{publisher}.")
    elif resource_type == "book_section":
        parts.append(f'"{title}."')
        editors = metadata.get("editors", [])
        book_title = metadata.get("book_title", "")
        publisher = metadata.get("publisher", "")
        pages = metadata.get("pages", "")
        if book_title:
            ed_part = f"edited by {', '.join(editors)}, " if editors else ""
            pg_part = f", {pages}" if pages else ""
            pub_year = f"{publisher}, {year}" if publisher and year else (publisher or year or "")
            parts.append(f"In {book_title}, {ed_part}{pg_part}. {pub_year}." if pub_year
                         else f"In {book_title}{pg_part}.")
    elif resource_type == "conference_paper":
        parts.append(f'"{title}."')
        conf = metadata.get("conference_name", "") or metadata.get("source", "")
        loc = metadata.get("conference_location", "")
        if conf:
            conf_part = f"Paper presented at {conf}"
            if loc:
                conf_part += f", {loc}"
            if year:
                conf_part += f", {year}"
            parts.append(conf_part + ".")
    elif resource_type == "thesis":
        degree = (metadata.get("degree_type", "") or "").lower()
        degree_label = (
            "master's thesis" if degree in ("masters", "master's", "master")
            else "doctoral dissertation"
        )
        institution = metadata.get("institution", "")
        parts.append(f'"{title}."')
        if institution and year:
            parts.append(f"{degree_label.capitalize()}, {institution}, {year}.")
        elif institution:
            parts.append(f"{degree_label.capitalize()}, {institution}.")
        elif year:
            parts.append(f"{degree_label.capitalize()}, {year}.")
    elif resource_type == "webpage":
        parts.append(f'"{title}."')
        source = metadata.get("source", "")
        accessed = metadata.get("accessed_date", "")
        if source:
            parts.append(f"{source}.")
        if accessed:
            parts.append(f"Accessed {accessed}.")
    elif resource_type == "software":
        version = metadata.get("version", "")
        ver_str = f". Version {version}" if version else ""
        parts.append(f"{title}{ver_str}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(f"{publisher}, {year}." if year else f"{publisher}.")
    elif resource_type == "report":
        report_number = metadata.get("report_number", "")
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        rn_str = f" Report No. {report_number}." if report_number else ""
        if publisher:
            parts.append(f"{publisher}, {year}.{rn_str}" if year else f"{publisher}.{rn_str}")
    elif resource_type == "dataset":
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(f"{publisher}, {year}." if year else f"{publisher}.")
    elif resource_type in ("film", "video", "audio"):
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(f"{publisher}, {year}." if year else f"{publisher}.")
    elif resource_type in ("patent", "presentation"):
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(f"{publisher}, {year}." if year else f"{publisher}.")
    else:
        parts.append(f"{title}.")
        publisher = metadata.get("publisher", "")
        if publisher:
            parts.append(f"{publisher}, {year}." if year else f"{publisher}.")

    doi = metadata.get("doi", "")
    url = metadata.get("url", "")
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)
    elif url:
        parts.append(url)

    return " ".join(parts)


def format_bibtex_entry(metadata: dict | None) -> str:
    """Generate BibTeX entry."""
    if not metadata:
        return "% Unable to generate citation: no metadata available."

    resource_type = metadata.get("resource_type", "other")

    _BIBTEX_TYPE = {
        "article": "article",
        "magazine_article": "article",
        "newspaper_article": "article",
        "book": "book",
        "book_section": "incollection",
        "conference_paper": "inproceedings",
        "report": "techreport",
        "dataset": "misc",
        "webpage": "misc",
        "film": "misc",
        "video": "misc",
        "audio": "misc",
        "patent": "misc",
        "presentation": "misc",
        "software": "misc",
    }

    authors = metadata.get("authors", [])
    first_author = authors[0].split("$$")[0].split(",")[0].strip() if authors else "unknown"
    first_author = "".join(c for c in first_author if c.isalnum())
    year = metadata.get("date", "")[:4] if metadata.get("date") else "YYYY"
    title_word = metadata.get("title", "untitled").split()[0] if metadata.get("title") else "untitled"
    title_word = "".join(c for c in title_word if c.isalnum())
    key = f"{first_author.lower()}{year}{title_word.lower()}"

    if resource_type == "thesis":
        degree = (metadata.get("degree_type", "") or "").lower()
        entry_type = "mastersthesis" if degree in ("masters", "master's", "master") else "phdthesis"
    else:
        entry_type = _BIBTEX_TYPE.get(resource_type, "misc")

    lines = [f"@{entry_type}{{{key},"]

    if authors:
        author_str = " and ".join([a.split("$$")[0].strip() for a in authors])
        lines.append(f"  author = {{{author_str}}},")

    title = metadata.get("title", "")
    if title:
        lines.append(f"  title = {{{title}}},")

    if year and year != "YYYY":
        lines.append(f"  year = {{{year}}},")

    if resource_type in ("article", "magazine_article", "newspaper_article"):
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
    elif resource_type == "book":
        publisher = metadata.get("publisher", "")
        if publisher:
            lines.append(f"  publisher = {{{publisher}}},")
        edition = metadata.get("edition", "")
        if edition:
            lines.append(f"  edition = {{{edition}}},")
    elif resource_type == "book_section":
        book_title = metadata.get("book_title", "")
        if book_title:
            lines.append(f"  booktitle = {{{book_title}}},")
        editors = metadata.get("editors", [])
        if editors:
            lines.append(f"  editor = {{{' and '.join(editors)}}},")
        publisher = metadata.get("publisher", "")
        if publisher:
            lines.append(f"  publisher = {{{publisher}}},")
        pages = metadata.get("pages", "")
        if pages:
            lines.append(f"  pages = {{{pages}}},")
    elif resource_type == "conference_paper":
        conf = metadata.get("conference_name", "") or metadata.get("source", "")
        if conf:
            lines.append(f"  booktitle = {{{conf}}},")
        loc = metadata.get("conference_location", "")
        if loc:
            lines.append(f"  address = {{{loc}}},")
        pages = metadata.get("pages", "")
        if pages:
            lines.append(f"  pages = {{{pages}}},")
    elif resource_type == "thesis":
        institution = metadata.get("institution", "")
        if institution:
            lines.append(f"  school = {{{institution}}},")
    elif resource_type == "report":
        report_number = metadata.get("report_number", "")
        if report_number:
            lines.append(f"  number = {{{report_number}}},")
        publisher = metadata.get("publisher", "")
        if publisher:
            lines.append(f"  institution = {{{publisher}}},")
    elif resource_type == "software":
        version = metadata.get("version", "")
        if version:
            lines.append(f"  version = {{{version}}},")
        publisher = metadata.get("publisher", "")
        if publisher:
            lines.append(f"  organization = {{{publisher}}},")
    else:
        publisher = metadata.get("publisher", "")
        if publisher:
            lines.append(f"  publisher = {{{publisher}}},")
        source = metadata.get("source", "")
        if source:
            lines.append(f"  howpublished = {{{source}}},")

    isbn = metadata.get("isbn", "")
    if isbn:
        lines.append(f"  isbn = {{{isbn}}},")

    issn = metadata.get("issn", "")
    if issn:
        lines.append(f"  issn = {{{issn}}},")

    doi = metadata.get("doi", "")
    if doi:
        lines.append(f"  doi = {{{doi}}},")

    url = metadata.get("url", "")
    if url and not doi:
        lines.append(f"  url = {{{url}}},")

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


def _parse_exlibris_link(raw_link: str) -> str:
    """Parse Ex Libris $$U delimited link format to extract clean URL.

    CDI records embed URLs in a format like:
      $$Uhttps://link.springer.com/content/pdf/...$$EPDF$$P50$$Gspringer$$H

    Extract the URL between $$U and the next $$ delimiter.
    If the link doesn't use this format, return it as-is.
    """
    if "$$U" in raw_link:
        # Extract URL after $$U, up to next $$ or end of string
        match = re.search(r'\$\$U(https?://[^\$]+)', raw_link)
        if match:
            return match.group(1)
    # Not in Ex Libris format — return as-is
    return raw_link


def extract_full_text_links(item: dict) -> dict | None:
    """Extract all full-text access links from a record.

    Parses Ex Libris $$U delimited format (Strategy C) to extract
    clean URLs from CDI record link fields.
    """
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
            result["html_links"].append(_parse_exlibris_link(link))

    for link in links.get("linktopdf", []):
        if isinstance(link, str):
            result["pdf_links"].append(_parse_exlibris_link(link))

    for link in links.get("linktorsrc", []):
        if isinstance(link, str):
            result["source_links"].append(_parse_exlibris_link(link))

    doi = addata.get("doi", [""])[0] if addata.get("doi") else ""
    if doi:
        if doi.startswith("http"):
            result["doi_url"] = doi
        else:
            result["doi_url"] = f"https://doi.org/{doi}"

    oa_indicators = links.get("openaccess", []) or links.get("openurl", [])
    result["open_access"] = len(oa_indicators) > 0

    return result
