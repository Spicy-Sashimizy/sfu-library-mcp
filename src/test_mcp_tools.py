"""
Test MCP tool calls directly.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from lib.tools import handle_tool_call
from lib.client import SFULibraryClient


def get_client():
    return SFULibraryClient(headless=True)


async def test_tools():
    print("=" * 60)
    print("Testing MCP Tools Directly")
    print("=" * 60)

    client = get_client()

    # Test get_token_status
    print("\n1. Testing get_token_status...")
    result = await handle_tool_call("get_token_status", {}, client)
    print(f"   Result: {result[0].text[:200]}...")

    # Test authenticate
    print("\n2. Testing authenticate...")
    result = await handle_tool_call("authenticate", {"force": False}, client)
    print(f"   Result: {result[0].text[:200]}...")

    # Test search_library
    print("\n3. Testing search_library...")
    result = await handle_tool_call("search_library", {
        "query": "artificial intelligence",
        "limit": 3
    }, client)
    print(f"   Result preview:\n{result[0].text[:500]}...")

    # Get a record ID for testing other tools
    print("\n4. Getting a record ID for citation tests...")
    result = await handle_tool_call("search_library", {
        "query": "machine learning mitchell",
        "limit": 1
    }, client)
    text = result[0].text
    record_id = None
    for line in text.split('\n'):
        if 'Record ID:' in line:
            record_id = line.split('Record ID:')[1].strip()
            break
    print(f"   Found record ID: {record_id}")

    if record_id:
        # Test generate_citation - APA
        print("\n5. Testing generate_citation (APA)...")
        result = await handle_tool_call("generate_citation", {
            "record_id": record_id,
            "format": "apa"
        }, client)
        print(f"   Result:\n{result[0].text}")

        # Test generate_citation - BibTeX
        print("\n6. Testing generate_citation (BibTeX)...")
        result = await handle_tool_call("generate_citation", {
            "record_id": record_id,
            "format": "bibtex"
        }, client)
        print(f"   Result:\n{result[0].text}")

        # Test get_full_text_links
        print("\n7. Testing get_full_text_links...")
        result = await handle_tool_call("get_full_text_links", {
            "record_id": record_id
        }, client)
        print(f"   Result:\n{result[0].text[:400]}...")

    # Test export_search_results - BibTeX
    print("\n8. Testing export_search_results (BibTeX)...")
    result = await handle_tool_call("export_search_results", {
        "query": "python programming",
        "format": "bibtex",
        "limit": 2
    }, client)
    print(f"   Result preview:\n{result[0].text[:500]}...")

    # Test export_search_results - CSV
    print("\n9. Testing export_search_results (CSV)...")
    result = await handle_tool_call("export_search_results", {
        "query": "data science",
        "format": "csv",
        "limit": 3
    }, client)
    print(f"   Result preview:\n{result[0].text[:500]}...")

    # Test batch_isbn_lookup
    print("\n10. Testing batch_isbn_lookup...")
    result = await handle_tool_call("batch_isbn_lookup", {
        "isbn_list": [
            "978-0-07-042807-0",
            "978-0-13-468599-1",
            "000-0-00-000000-0"
        ]
    }, client)
    print(f"   Result:\n{result[0].text}")

    print("\n" + "=" * 60)
    print("All MCP tool tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_tools())
