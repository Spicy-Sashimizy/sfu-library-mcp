"""Result formatting for search results and item details.

Extracted from the monolith. Implements DATA-004: character encoding handling.
"""

import logging

from lib.validators import normalize_encoding

logger = logging.getLogger("sfu_library_mcp")


def format_search_results(results: dict | None, metadata: dict | None = None) -> str:
    """Format search results for display.

    Args:
        results: Primo API response dict with docs and info.
        metadata: Optional dict with query metadata for the search hints footer.
            Keys: query, field, sort, resource_type.
    """
    if not results:
        logger.debug("format_search_results called with empty results")
        return "No results found or search failed."

    docs = results.get("docs", [])
    info = results.get("info", {})
    total = info.get("total", 0)
    logger.debug("Formatting %d docs from %d total results", len(docs), total)

    output = [f"Found {total:,} total results\n"]
    output.append("=" * 60 + "\n")

    all_subjects: list[str] = []
    electronic_count = 0

    for i, doc in enumerate(docs, 1):
        pnx = doc.get("pnx", {})
        display = pnx.get("display", {})
        control = pnx.get("control", {})
        addata = pnx.get("addata", {})
        links = pnx.get("links", {})
        delivery = pnx.get("delivery", {})

        title = normalize_encoding(display.get("title", ["No title"])[0])
        creators = display.get("creator", display.get("contributor", []))
        creator = normalize_encoding(creators[0]) if creators else "Unknown"
        pub_date = display.get("creationdate", ["N/A"])[0]
        doc_type = display.get("type", ["N/A"])[0]
        description = normalize_encoding(display.get("description", [""])[0][:300])
        source = normalize_encoding(display.get("source", [""])[0]) if display.get("source") else ""
        publisher = normalize_encoding(display.get("publisher", [""])[0]) if display.get("publisher") else ""

        doc_id = control.get("recordid", [""])[0] if control.get("recordid") else ""
        isbn = addata.get("isbn", [""])[0] if addata.get("isbn") else ""
        issn = addata.get("issn", [""])[0] if addata.get("issn") else ""
        doi = addata.get("doi", [""])[0] if addata.get("doi") else ""

        fulltext_links = links.get("linktorsrc", []) or links.get("linktohtml", [])
        availability = delivery.get("availability", [""])[0] if delivery.get("availability") else ""

        # Extract subject headings (top 3 per result)
        subjects = display.get("subject", [])
        top_subjects = [s.split("$$")[0] for s in subjects[:3]]
        all_subjects.extend(top_subjects)

        # Track electronic availability
        if fulltext_links or "online" in availability.lower() or "available" in availability.lower():
            electronic_count += 1

        output.append(f"{i}. {title}\n")
        output.append(f"   Author: {creator}\n")
        output.append(f"   Date: {pub_date}\n")
        output.append(f"   Type: {doc_type}\n")

        if top_subjects:
            output.append(f"   Subjects: {', '.join(top_subjects)}\n")

        if source:
            output.append(f"   Source: {source}\n")
        if publisher:
            output.append(f"   Publisher: {publisher}\n")
        if isbn:
            output.append(f"   ISBN: {isbn}\n")
        if issn:
            output.append(f"   ISSN: {issn}\n")
        if doi:
            output.append(f"   DOI: {doi}\n")
        if doc_id:
            output.append(f"   Record ID: {doc_id}\n")
        if availability:
            output.append(f"   Availability: {availability}\n")
        if description:
            output.append(f"   Description: {description}...\n")
        if fulltext_links:
            output.append(f"   Full Text: Available\n")

        output.append("\n")

    # Search metadata footer
    if docs:
        output.append("--- Search Metadata ---\n")
        if metadata:
            query_str = metadata.get("query", "")
            field_str = metadata.get("field", "any")
            sort_str = metadata.get("sort", "rank")
            output.append(f"Query: {query_str} | Field: {field_str} | Sort: {sort_str}\n")

        # Aggregate unique subjects (top 5)
        seen: set[str] = set()
        unique_subjects: list[str] = []
        for s in all_subjects:
            s_lower = s.lower()
            if s_lower not in seen and s:
                seen.add(s_lower)
                unique_subjects.append(s)
            if len(unique_subjects) >= 5:
                break
        if unique_subjects:
            output.append(f"Top subjects across results: {', '.join(unique_subjects)}\n")
        output.append(f"Electronic resources available: {electronic_count}\n")

    return "".join(output)


def format_item_details(item: dict | None) -> str:
    """Format detailed item information."""
    if not item:
        logger.debug("format_item_details called with empty item")
        return "Could not retrieve item details."

    pnx = item.get("pnx", {})
    record_id = pnx.get("control", {}).get("recordid", [""])[0] if pnx.get("control", {}).get("recordid") else "unknown"
    logger.debug("Formatting item details for record: %s", record_id)
    display = pnx.get("display", {})
    control = pnx.get("control", {})
    addata = pnx.get("addata", {})
    links = pnx.get("links", {})
    delivery = pnx.get("delivery", {})

    output = ["=" * 60 + "\n"]
    output.append("ITEM DETAILS\n")
    output.append("=" * 60 + "\n\n")

    title = normalize_encoding(display.get("title", ["No title"])[0])
    output.append(f"Title: {title}\n\n")

    creators = display.get("creator", [])
    if creators:
        output.append(f"Author(s): {', '.join(normalize_encoding(c) for c in creators)}\n")

    contributors = display.get("contributor", [])
    if contributors:
        output.append(f"Contributor(s): {', '.join(normalize_encoding(c) for c in contributors)}\n")

    pub_date = display.get("creationdate", [""])[0]
    if pub_date:
        output.append(f"Publication Date: {pub_date}\n")

    doc_type = display.get("type", [""])[0]
    if doc_type:
        output.append(f"Type: {doc_type}\n")

    publisher = display.get("publisher", [""])[0]
    if publisher:
        output.append(f"Publisher: {normalize_encoding(publisher)}\n")

    source = display.get("source", [""])[0]
    if source:
        output.append(f"Source/Journal: {normalize_encoding(source)}\n")

    output.append("\n--- Identifiers ---\n")
    isbn = addata.get("isbn", [])
    if isbn:
        output.append(f"ISBN: {', '.join(isbn)}\n")

    issn = addata.get("issn", [])
    if issn:
        output.append(f"ISSN: {', '.join(issn)}\n")

    doi = addata.get("doi", [])
    if doi:
        output.append(f"DOI: {', '.join(doi)}\n")

    doc_id = control.get("recordid", [""])[0]
    if doc_id:
        output.append(f"Record ID: {doc_id}\n")

    description = display.get("description", [])
    if description:
        output.append("\n--- Description ---\n")
        for desc in description:
            output.append(f"{normalize_encoding(desc)}\n")

    subjects = display.get("subject", [])
    if subjects:
        output.append("\n--- Subjects ---\n")
        for subj in subjects[:10]:
            output.append(f"- {subj}\n")

    output.append("\n--- Access Links ---\n")

    linktohtml = links.get("linktohtml", [])
    if linktohtml:
        output.append("HTML Link: Available\n")

    linktorsrc = links.get("linktorsrc", [])
    if linktorsrc:
        output.append("Full Text Link: Available\n")

    linktopdf = links.get("linktopdf", [])
    if linktopdf:
        output.append("PDF Link: Available\n")

    availability = delivery.get("availability", [])
    if availability:
        output.append(f"\nAvailability: {', '.join(availability)}\n")

    holding = delivery.get("holding", [])
    if holding:
        output.append("\n--- Holdings ---\n")
        for h in holding[:5]:
            lib = h.get("libraryCode", "")
            location = h.get("subLocationCode", "")
            call = h.get("callNumber", "")
            output.append(f"  {lib} - {location}: {call}\n")

    return "".join(output)
