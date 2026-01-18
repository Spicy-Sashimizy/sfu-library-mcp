"""
Test MCP tool calls directly.

================================================================================
TODO: MCP TOOL TESTING IMPROVEMENTS TRACKING
================================================================================
SECTION: MCP TOOL TEST COVERAGE
--------------------------------
[ ] TODO-MCP-001: Add test for invalid tool name handling
[ ] TODO-MCP-002: Add test for missing required parameters
[ ] TODO-MCP-003: Add test for invalid parameter types
[ ] TODO-MCP-004: Add test for citation edge cases (missing metadata)
[ ] TODO-MCP-005: Add test for batch operation failures
[ ] TODO-MCP-006: Add test for ISBN validation edge cases
[ ] TODO-MCP-007: Add test for export format conversions
[ ] TODO-MCP-008: Add test for concurrent tool calls

SECTION: ASYNC OPERATION TESTING
---------------------------------
[ ] TODO-ASYNC-001: Add test for timeout handling on async operations
[ ] TODO-ASYNC-002: Add test for cancellation of pending requests
[ ] TODO-ASYNC-003: Add test for error propagation in async context

SECTION: DATA VALIDATION
-------------------------
[ ] TODO-VAL-001: Validate record ID format before API calls
[ ] TODO-VAL-002: Verify citation format compliance with standards
[ ] TODO-VAL-003: Validate ISBN checksums before lookup
[ ] TODO-VAL-004: Check for data truncation in export formats

SECTION: TEST INFRASTRUCTURE
-----------------------------
[ ] TODO-INFRA-001: Add command-line arguments for test selection
[ ] TODO-INFRA-002: Implement test result aggregation and reporting
[ ] TODO-INFRA-003: Add mock API server for offline testing
[ ] TODO-INFRA-004: Add test data fixtures for reproducible runs
[ ] TODO-INFRA-005: Implement test parallelization

SECTION: ERROR SCENARIO TESTING
--------------------------------
[ ] TODO-ERR-001: Test behavior when API returns 500 errors
[ ] TODO-ERR-002: Test behavior when API rate limits are hit
[ ] TODO-ERR-003: Test behavior with malformed JSON responses
[ ] TODO-ERR-004: Test behavior with network timeouts

================================================================================
END OF TODO LIST - Last updated: 2026-01-11
================================================================================
"""

import asyncio
import sys
sys.path.insert(0, '.')

from sfu_library_mcp_server import call_tool, get_client

# TODO-INFRA-002: Add pass/fail counting and summary reporting
# TODO-ASYNC-001: Add timeout wrapper for entire test suite
async def test_tools():
    print("=" * 60)
    print("Testing MCP Tools Directly")
    print("=" * 60)

    # TODO-MCP-001: No test for invalid tool names
    # Test get_token_status
    print("\n1. Testing get_token_status...")
    result = await call_tool("get_token_status", {})
    print(f"   Result: {result[0].text[:200]}...")

    # TODO-MCP-002: No test for missing/invalid parameters
    # Test authenticate
    print("\n2. Testing authenticate...")
    result = await call_tool("authenticate", {"force": False})
    print(f"   Result: {result[0].text[:200]}...")

    # Test search_library
    print("\n3. Testing search_library...")
    result = await call_tool("search_library", {
        "query": "artificial intelligence",
        "limit": 3
    })
    print(f"   Result preview:\n{result[0].text[:500]}...")

    # Get a record ID for testing other tools
    # Search for a specific book to get a record ID
    print("\n4. Getting a record ID for citation tests...")
    result = await call_tool("search_library", {
        "query": "machine learning mitchell",
        "limit": 1
    })
    # Extract record ID from result
    text = result[0].text
    record_id = None
    for line in text.split('\n'):
        if 'Record ID:' in line:
            record_id = line.split('Record ID:')[1].strip()
            break
    print(f"   Found record ID: {record_id}")

    # TODO-VAL-001: No validation of record_id format before use
    # TODO-MCP-004: No test for citation with missing metadata
    if record_id:
        # Test generate_citation - APA
        print("\n5. Testing generate_citation (APA)...")
        result = await call_tool("generate_citation", {
            "record_id": record_id,
            "format": "apa"
        })
        print(f"   Result:\n{result[0].text}")

        # Test generate_citation - BibTeX
        print("\n6. Testing generate_citation (BibTeX)...")
        result = await call_tool("generate_citation", {
            "record_id": record_id,
            "format": "bibtex"
        })
        print(f"   Result:\n{result[0].text}")

        # Test get_full_text_links
        print("\n7. Testing get_full_text_links...")
        result = await call_tool("get_full_text_links", {
            "record_id": record_id
        })
        print(f"   Result:\n{result[0].text[:400]}...")

    # Test export_search_results - BibTeX
    print("\n8. Testing export_search_results (BibTeX)...")
    result = await call_tool("export_search_results", {
        "query": "python programming",
        "format": "bibtex",
        "limit": 2
    })
    print(f"   Result preview:\n{result[0].text[:500]}...")

    # Test export_search_results - CSV
    print("\n9. Testing export_search_results (CSV)...")
    result = await call_tool("export_search_results", {
        "query": "data science",
        "format": "csv",
        "limit": 3
    })
    print(f"   Result preview:\n{result[0].text[:500]}...")

    # Test batch_isbn_lookup
    print("\n10. Testing batch_isbn_lookup...")
    # TODO-VAL-003: No ISBN checksum validation test
    # TODO-MCP-005: No test for batch operation with partial failures
    result = await call_tool("batch_isbn_lookup", {
        "isbn_list": [
            "978-0-07-042807-0",  # Machine Learning by Mitchell
            "978-0-13-468599-1",  # Clean Code
            "000-0-00-000000-0"   # Invalid ISBN
        ]
    })
    print(f"   Result:\n{result[0].text}")

    print("\n" + "=" * 60)
    print("All MCP tool tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_tools())
