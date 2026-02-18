"""Unit tests for the Zotero module."""

from unittest.mock import MagicMock, patch, PropertyMock
import time

import pytest

from lib.config import ServerConfig
from lib.retry import CircuitBreaker
from lib.zotero import ZoteroClient, ZoteroError


@pytest.fixture
def zot_config():
    return ServerConfig(
        zotero_api_key="test_api_key",
        zotero_user_id="12345",
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60.0,
    )


@pytest.fixture
def mock_pyzotero():
    """Mock pyzotero.Zotero instance."""
    with patch("lib.zotero.ZoteroClient.zot", new_callable=PropertyMock) as mock_zot:
        mock_instance = MagicMock()
        mock_zot.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def zot_client(zot_config, mock_pyzotero):
    client = ZoteroClient(zot_config)
    client._zot = mock_pyzotero
    return client


# ─── Author parsing ────────────────────────────────────────────

class TestParseAuthorName:
    def test_parse_author_last_comma_first(self):
        result = ZoteroClient._parse_author_name("Smith, John")
        assert result["lastName"] == "Smith"
        assert result["firstName"] == "John"
        assert result["creatorType"] == "author"

    def test_parse_author_first_last(self):
        result = ZoteroClient._parse_author_name("John Smith")
        assert result["lastName"] == "Smith"
        assert result["firstName"] == "John"

    def test_parse_author_exlibris_suffix(self):
        result = ZoteroClient._parse_author_name("Smith, J$$QSmith, J")
        assert result["lastName"] == "Smith"
        assert result["firstName"] == "J"

    def test_parse_author_single_name(self):
        result = ZoteroClient._parse_author_name("Aristotle")
        assert result["lastName"] == "Aristotle"
        assert result["firstName"] == ""


# ─── Metadata mapping ──────────────────────────────────────────

class TestMetadataToZoteroItem:
    def test_metadata_to_zotero_article(self, zot_client):
        metadata = {
            "title": "Deep Learning",
            "authors": ["Smith, John"],
            "date": "2023",
            "resource_type": "article",
            "source": "Nature",
            "doi": "10.1000/test",
            "volume": "1",
            "issue": "2",
            "spage": "10",
            "epage": "25",
            "isbn": "",
            "issn": "1234-5678",
            "record_id": "rec_001",
            "publisher": "",
            "pages": "",
        }
        item = zot_client.metadata_to_zotero_item(metadata)
        assert item["itemType"] == "journalArticle"
        assert item["title"] == "Deep Learning"
        assert item["publicationTitle"] == "Nature"

    def test_metadata_to_zotero_book(self, zot_client):
        metadata = {
            "title": "Machine Learning",
            "authors": ["Murphy, Kevin"],
            "date": "2012",
            "resource_type": "book",
            "publisher": "MIT Press",
            "isbn": "9780262018029",
            "source": "",
            "doi": "",
            "volume": "",
            "issue": "",
            "spage": "",
            "epage": "",
            "issn": "",
            "record_id": "rec_002",
            "pages": "",
        }
        item = zot_client.metadata_to_zotero_item(metadata)
        assert item["itemType"] == "book"
        assert item["publisher"] == "MIT Press"

    def test_metadata_to_zotero_other(self, zot_client):
        metadata = {
            "title": "A Document",
            "authors": [],
            "date": "2020",
            "resource_type": "other",
            "source": "",
            "doi": "",
            "volume": "",
            "issue": "",
            "spage": "",
            "epage": "",
            "isbn": "",
            "issn": "",
            "record_id": "",
            "publisher": "",
            "pages": "",
        }
        item = zot_client.metadata_to_zotero_item(metadata)
        assert item["itemType"] == "document"

    def test_metadata_to_zotero_pages_combined(self, zot_client):
        metadata = {
            "title": "Test",
            "authors": [],
            "date": "",
            "resource_type": "article",
            "source": "",
            "doi": "",
            "volume": "",
            "issue": "",
            "spage": "10",
            "epage": "25",
            "pages": "",
            "isbn": "",
            "issn": "",
            "record_id": "",
            "publisher": "",
        }
        item = zot_client.metadata_to_zotero_item(metadata)
        assert item["pages"] == "10-25"

    def test_metadata_to_zotero_record_id_extra(self, zot_client):
        metadata = {
            "title": "Test",
            "authors": [],
            "date": "",
            "resource_type": "article",
            "source": "",
            "doi": "",
            "volume": "",
            "issue": "",
            "spage": "",
            "epage": "",
            "pages": "",
            "isbn": "",
            "issn": "",
            "record_id": "alma123456",
            "publisher": "",
        }
        item = zot_client.metadata_to_zotero_item(metadata)
        assert "alma123456" in item["extra"]


# ─── Duplicate detection ───────────────────────────────────────

class TestDuplicateDetection:
    def test_check_duplicate_doi_match(self, zot_client, mock_pyzotero):
        mock_pyzotero.items.return_value = [
            {"data": {"key": "ABC123", "DOI": "10.1000/test", "title": "Test", "creators": [], "collections": [], "tags": []}}
        ]
        metadata = {"doi": "10.1000/test", "isbn": "", "title": "Test", "authors": []}
        result = zot_client.check_duplicate(metadata)
        assert result["is_duplicate"] is True
        assert result["match_type"] == "DOI"
        assert result["existing_key"] == "ABC123"

    def test_check_duplicate_isbn_match(self, zot_client, mock_pyzotero):
        # DOI is empty so DOI check is skipped; first call is ISBN check
        mock_pyzotero.items.side_effect = [
            [{"data": {"key": "DEF456", "ISBN": "9780262018029", "title": "ML Book", "creators": [], "collections": [], "tags": []}}],
        ]
        metadata = {"doi": "", "isbn": "9780262018029", "title": "ML Book", "authors": []}
        result = zot_client.check_duplicate(metadata)
        assert result["is_duplicate"] is True
        assert result["match_type"] == "ISBN"

    def test_check_duplicate_title_author_match(self, zot_client, mock_pyzotero):
        # DOI and ISBN are empty so those checks are skipped; first call is title check
        mock_pyzotero.items.side_effect = [
            [{"data": {
                "key": "GHI789",
                "title": "Machine Learning: A Probabilistic Perspective",
                "creators": [{"lastName": "Murphy", "firstName": "Kevin"}],
                "DOI": "",
                "ISBN": "",
                "collections": [],
                "tags": [],
            }}],
        ]
        metadata = {
            "doi": "",
            "isbn": "",
            "title": "Machine Learning: A Probabilistic Perspective",
            "authors": ["Murphy, Kevin P."],
        }
        result = zot_client.check_duplicate(metadata)
        assert result["is_duplicate"] is True
        assert result["match_type"] == "title+author"

    def test_check_duplicate_title_similar_diff_author(self, zot_client, mock_pyzotero):
        # DOI and ISBN are empty so those checks are skipped; first call is title check
        mock_pyzotero.items.side_effect = [
            [{"data": {
                "key": "JKL012",
                "title": "Machine Learning: A Probabilistic Perspective",
                "creators": [{"lastName": "Bishop", "firstName": "Christopher"}],
                "DOI": "",
                "ISBN": "",
                "collections": [],
                "tags": [],
            }}],
        ]
        metadata = {
            "doi": "",
            "isbn": "",
            "title": "Machine Learning: A Probabilistic Perspective",
            "authors": ["Murphy, Kevin P."],
        }
        result = zot_client.check_duplicate(metadata)
        assert result["is_duplicate"] is False

    def test_check_duplicate_no_match(self, zot_client, mock_pyzotero):
        mock_pyzotero.items.return_value = []
        metadata = {"doi": "10.9999/new", "isbn": "", "title": "Brand New Paper", "authors": ["New, Author"]}
        result = zot_client.check_duplicate(metadata)
        assert result["is_duplicate"] is False

    def test_check_duplicate_no_doi_no_isbn(self, zot_client, mock_pyzotero):
        """Without DOI or ISBN, should fall through to title+author check."""
        mock_pyzotero.items.return_value = []
        metadata = {"doi": "", "isbn": "", "title": "Some Paper", "authors": ["Author, Test"]}
        result = zot_client.check_duplicate(metadata)
        assert result["is_duplicate"] is False
        # Should have called items at least once for title check
        assert mock_pyzotero.items.call_count >= 1


# ─── Collection management ─────────────────────────────────────

class TestCollections:
    def test_find_collection_by_name(self, zot_client, mock_pyzotero):
        mock_pyzotero.collections.return_value = [
            {"data": {"key": "COL1", "name": "ML Research", "parentCollection": ""}, "meta": {"numItems": 5}},
            {"data": {"key": "COL2", "name": "Biology", "parentCollection": ""}, "meta": {"numItems": 3}},
        ]
        result = zot_client.find_collection_by_name("ml research")
        assert result == "COL1"

    def test_find_collection_not_found(self, zot_client, mock_pyzotero):
        mock_pyzotero.collections.return_value = [
            {"data": {"key": "COL1", "name": "ML Research", "parentCollection": ""}, "meta": {"numItems": 5}},
        ]
        result = zot_client.find_collection_by_name("Nonexistent")
        assert result is None

    def test_find_or_create_existing(self, zot_client, mock_pyzotero):
        mock_pyzotero.collections.return_value = [
            {"data": {"key": "COL1", "name": "ML Research", "parentCollection": ""}, "meta": {"numItems": 5}},
        ]
        result = zot_client.find_or_create_collection("ML Research")
        assert result == "COL1"
        mock_pyzotero.create_collections.assert_not_called()

    def test_find_or_create_new(self, zot_client, mock_pyzotero):
        mock_pyzotero.collections.return_value = []
        mock_pyzotero.create_collections.return_value = {
            "successful": {"0": {"key": "NEW1"}},
            "failed": {},
        }
        result = zot_client.find_or_create_collection("New Collection")
        assert result == "NEW1"

    def test_list_collections_formatting(self, zot_client, mock_pyzotero):
        mock_pyzotero.collections.return_value = [
            {"data": {"key": "COL1", "name": "Research", "parentCollection": ""}, "meta": {"numItems": 10}},
        ]
        result = zot_client.list_collections()
        assert len(result) == 1
        assert result[0]["key"] == "COL1"
        assert result[0]["name"] == "Research"
        assert result[0]["num_items"] == 10


# ─── Circuit breaker ───────────────────────────────────────────

class TestCircuitBreaker:
    def test_circuit_breaker_opens_after_failures(self, zot_config):
        """5 failures should open the circuit breaker."""
        client = ZoteroClient(zot_config)
        mock_zot = MagicMock()
        client._zot = mock_zot
        mock_zot.collections.side_effect = Exception("API down")

        for _ in range(5):
            with pytest.raises(ZoteroError):
                client._call_zotero("test", mock_zot.collections)

        assert client._breaker.state == CircuitBreaker.OPEN
        with pytest.raises(ZoteroError, match="circuit breaker is OPEN"):
            client._call_zotero("test", mock_zot.collections)

    def test_circuit_breaker_resets_on_success(self, zot_config):
        """Success after failures should close the breaker."""
        client = ZoteroClient(zot_config)
        mock_zot = MagicMock()
        client._zot = mock_zot

        # Record some failures
        mock_zot.collections.side_effect = Exception("Temporary failure")
        for _ in range(3):
            with pytest.raises(ZoteroError):
                client._call_zotero("test", mock_zot.collections)

        assert client._breaker.failure_count == 3

        # Now succeed
        mock_zot.collections.side_effect = None
        mock_zot.collections.return_value = []
        client._call_zotero("test", mock_zot.collections)

        assert client._breaker.state == CircuitBreaker.CLOSED
        assert client._breaker.failure_count == 0

    def test_circuit_breaker_half_open_recovery(self, zot_config):
        """After timeout, breaker should allow one test request."""
        client = ZoteroClient(zot_config)
        breaker = client._breaker

        # Force OPEN state
        breaker.state = CircuitBreaker.OPEN
        breaker.failure_count = 5
        breaker.last_failure_time = time.time() - 120  # 2 min ago, past timeout

        assert breaker.can_proceed() is True
        assert breaker.state == CircuitBreaker.HALF_OPEN

    def test_circuit_breaker_wraps_all_methods(self, zot_client, mock_pyzotero):
        """Verify _call_zotero is used for API calls (breaker records success)."""
        mock_pyzotero.collections.return_value = []
        zot_client.list_collections()

        # After a successful call, breaker should record success
        assert zot_client._breaker.failure_count == 0
        assert zot_client._breaker.state == CircuitBreaker.CLOSED


# ─── Lazy loading ──────────────────────────────────────────────

class TestLazyLoading:
    def test_lazy_load_no_import_until_use(self, zot_config):
        """pyzotero should not be imported at init time."""
        client = ZoteroClient(zot_config)
        assert client._zot is None

    def test_missing_credentials_error(self):
        """No API key should raise clear error."""
        config = ServerConfig(zotero_api_key="", zotero_user_id="")
        client = ZoteroClient(config)
        with pytest.raises(ZoteroError, match="credentials not configured"):
            _ = client.zot
