"""
Test script for SFU Library MCP client.
Tests authentication and search functionality.
"""

import sys
sys.path.insert(0, '.')

from lib.client import SFULibraryClient
from lib.formatters import format_search_results, format_item_details


def test_client():
    print("=" * 60)
    print("SFU Library MCP Client Test")
    print("=" * 60)

    # Create client
    client = SFULibraryClient(headless=True)

    # Check token status first
    print("\n1. Checking token status...")
    status = client.get_token_status()
    print(f"   Status: {status}")

    # Authenticate
    print("\n2. Authenticating...")
    if client.ensure_authenticated():
        print("   Authentication successful!")
        status = client.get_token_status()
        print(f"   User: {status.get('user')} ({status.get('userId')})")
        print(f"   Expires: {status.get('expiresIn')}")
    else:
        print("   Authentication FAILED!")
        return False

    # Test search
    print("\n3. Testing search: 'machine learning'...")
    results = client.search("machine learning", limit=5)
    if results:
        print(f"   Found {results.get('info', {}).get('total', 0)} results")
        print("\n" + format_search_results(results))
    else:
        print("   Search FAILED!")
        return False

    # Test search by author
    print("\n4. Testing author search: 'Einstein'...")
    results = client.search("Einstein", limit=3, field='creator')
    if results:
        print(f"   Found {results.get('info', {}).get('total', 0)} results")
        docs = results.get('docs', [])
        for i, doc in enumerate(docs[:3], 1):
            title = doc.get('pnx', {}).get('display', {}).get('title', ['No title'])[0]
            print(f"   {i}. {title[:60]}...")

    # Test electronic resources
    print("\n5. Testing electronic resources search: 'python'...")
    results = client.search("python", limit=3, tab="online_only_tab", scope="ElectronicOnly_scope")
    if results:
        print(f"   Found {results.get('info', {}).get('total', 0)} electronic resources")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_client()
    sys.exit(0 if success else 1)
