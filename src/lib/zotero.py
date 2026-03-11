"""Zotero API wrapper for saving and browsing library items.

Provides ZoteroClient that maps PNX metadata to Zotero items, manages
collections, detects duplicates via multi-signal deconfliction, and
wraps all API calls with a circuit breaker.
"""

import logging
import os
import re

from lib.config import ServerConfig
from lib.retry import CircuitBreaker

logger = logging.getLogger("sfu_library_mcp")


class ZoteroError(Exception):
    """Raised when a Zotero API operation fails."""
    pass


class ZoteroClient:
    """Zotero API client with circuit breaker and collection management."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._zot = None
        self._breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout,
        )
        self._authenticated = False
        self._auth_info = {}

    @property
    def zot(self):
        """Lazy-load pyzotero on first use."""
        if self._zot is None:
            if not self.config.zotero_api_key or not self.config.zotero_user_id:
                raise ZoteroError(
                    "Zotero credentials not configured. Set SFU_ZOTERO_API_KEY "
                    "and SFU_ZOTERO_USER_ID environment variables."
                )
            from pyzotero import zotero
            self._zot = zotero.Zotero(
                self.config.zotero_user_id,
                "user",
                self.config.zotero_api_key,
            )
        return self._zot

    def _call_zotero(self, operation_name: str, func, *args, **kwargs):
        """Wrapper for all Zotero API calls with circuit breaker + logging."""
        if not self._breaker.can_proceed():
            raise ZoteroError(
                f"Zotero API circuit breaker is OPEN — too many recent failures. "
                f"Will retry automatically after {self.config.circuit_breaker_timeout}s."
            )
        try:
            result = func(*args, **kwargs)
            self._breaker.record_success()
            logger.info("Zotero API: %s succeeded", operation_name)
            return result
        except ZoteroError:
            self._breaker.record_failure()
            raise
        except Exception as e:
            self._breaker.record_failure()
            logger.error(
                "Zotero API: %s failed (%s), failure_count=%d",
                operation_name, e, self._breaker.failure_count,
            )
            raise ZoteroError(f"Zotero API error in {operation_name}: {e}") from e

    # ─── Authentication ─────────────────────────────────────────────

    def verify_credentials(self) -> dict:
        """Verify Zotero API credentials by calling key_info().

        Returns:
            dict with "valid" bool and credential details on success,
            or "valid": False with "message" on failure.
        """
        if not self.config.zotero_api_key or not self.config.zotero_user_id:
            return {
                "valid": False,
                "message": "Zotero credentials not configured. "
                           "Set SFU_ZOTERO_API_KEY and SFU_ZOTERO_USER_ID.",
            }
        try:
            info = self._call_zotero("key_info", self.zot.key_info)
        except ZoteroError as e:
            self._authenticated = False
            self._auth_info = {}
            return {"valid": False, "message": str(e)}

        # Parse access permissions from key_info response
        access = info.get("access", {})
        user_access = access.get("user", {})
        username = info.get("username", "")
        user_id = info.get("userID", self.config.zotero_user_id)

        self._authenticated = True
        self._auth_info = {
            "username": username,
            "userID": user_id,
            "access": {
                "library": user_access.get("library", False),
                "files": user_access.get("files", False),
                "notes": user_access.get("notes", False),
                "write": user_access.get("write", False),
            },
        }
        logger.info("Zotero credentials verified for user: %s (ID: %s)", username, user_id)
        return {
            "valid": True,
            "username": username,
            "userID": user_id,
            "access": self._auth_info["access"],
        }

    def get_auth_status(self) -> dict:
        """Return cached Zotero auth state (no API call).

        Returns:
            dict with credentials_configured, validated, and cached info.
        """
        credentials_configured = bool(
            self.config.zotero_api_key and self.config.zotero_user_id
        )
        if not self._authenticated:
            return {
                "credentials_configured": credentials_configured,
                "validated": False,
                "message": "Not yet verified. Call verify_credentials() or ensure_authenticated().",
            }
        return {
            "credentials_configured": credentials_configured,
            "validated": True,
            "username": self._auth_info.get("username", ""),
            "userID": self._auth_info.get("userID", ""),
            "access": self._auth_info.get("access", {}),
        }

    def ensure_authenticated(self, force: bool = False) -> bool:
        """Ensure Zotero credentials are valid (lazy, cached).

        Args:
            force: If True, re-verify even if already authenticated.

        Returns:
            True if credentials are valid, False otherwise.
        """
        if force:
            self._authenticated = False
            self._auth_info = {}

        if self._authenticated:
            return True

        if not self.config.zotero_api_key or not self.config.zotero_user_id:
            return False

        result = self.verify_credentials()
        return result["valid"]

    # ─── Author parsing ────────────────────────────────────────────

    @staticmethod
    def _parse_author_name(author_str: str) -> dict:
        """Parse an author string into Zotero creator dict.

        Handles:
        - "Last, First" format
        - "First Last" format
        - Ex Libris "Smith, J$$QSmith, J" suffix format
        - Single name (e.g. "Aristotle")
        """
        if not author_str:
            return {"creatorType": "author", "lastName": "", "firstName": ""}

        # Strip Ex Libris $$Q suffix
        if "$$Q" in author_str:
            author_str = author_str.split("$$Q")[0].strip()
        # Also handle $$D display variants
        if "$$" in author_str:
            author_str = author_str.split("$$")[0].strip()

        if "," in author_str:
            parts = author_str.split(",", 1)
            return {
                "creatorType": "author",
                "lastName": parts[0].strip(),
                "firstName": parts[1].strip(),
            }
        else:
            parts = author_str.strip().split()
            if len(parts) == 1:
                return {
                    "creatorType": "author",
                    "lastName": parts[0],
                    "firstName": "",
                }
            return {
                "creatorType": "author",
                "lastName": parts[-1],
                "firstName": " ".join(parts[:-1]),
            }

    # ─── Metadata mapping ──────────────────────────────────────────

    def metadata_to_zotero_item(self, metadata: dict) -> dict:
        """Map extract_metadata() output to a Zotero item template."""
        resource_type = metadata.get("resource_type", "other")
        type_map = {
            "article": "journalArticle",
            "book": "book",
            "other": "document",
        }
        item_type = type_map.get(resource_type, "document")

        authors = metadata.get("authors") or metadata.get("creators", [])
        creators = [self._parse_author_name(a) for a in authors]

        item = {
            "itemType": item_type,
            "title": metadata.get("title", ""),
            "creators": creators,
            "date": metadata.get("date", ""),
            "DOI": metadata.get("doi", ""),
            "ISBN": metadata.get("isbn", ""),
            "ISSN": metadata.get("issn", ""),
            "volume": metadata.get("volume", ""),
            "issue": metadata.get("issue", ""),
            "extra": "",
        }

        # Pages
        spage = metadata.get("spage", "")
        epage = metadata.get("epage", "")
        pages = metadata.get("pages", "")
        if pages:
            item["pages"] = pages
        elif spage and epage:
            item["pages"] = f"{spage}-{epage}"
        elif spage:
            item["pages"] = spage

        # Type-specific fields
        if item_type == "journalArticle":
            item["publicationTitle"] = metadata.get("source", "")
        elif item_type == "book":
            item["publisher"] = metadata.get("publisher", "")
        else:
            item["publicationTitle"] = metadata.get("source", "")
            item["publisher"] = metadata.get("publisher", "")

        # Traceability
        record_id = metadata.get("record_id", "")
        if record_id:
            item["extra"] = f"SFU-Library-RecordID: {record_id}"

        return item

    # ─── CRUD operations ───────────────────────────────────────────

    def create_item(self, zotero_item: dict, collection_key: str | None = None) -> str:
        """Create a Zotero item, optionally in a collection. Returns item key."""
        if collection_key:
            zotero_item["collections"] = [collection_key]

        resp = self._call_zotero(
            "create_item",
            self.zot.create_items,
            [zotero_item],
        )

        if not resp or "successful" not in resp:
            raise ZoteroError(f"Unexpected response from create_items: {resp}")

        successful = resp.get("successful", {})
        if not successful:
            failed = resp.get("failed", {})
            raise ZoteroError(f"Failed to create Zotero item: {failed}")

        item_key = successful["0"]["key"]
        logger.info("Created Zotero item: %s (%s)", item_key, zotero_item.get("title", "")[:60])
        return item_key

    def attach_pdf(self, item_key: str, pdf_path: str) -> None:
        """Attach a PDF file to a Zotero item."""
        self._call_zotero(
            "attach_pdf",
            self.zot.attachment_simple,
            [pdf_path],
            item_key,
        )
        logger.info("Attached PDF to Zotero item %s: %s", item_key, pdf_path)

    # ─── Collection management ─────────────────────────────────────

    def list_collections(self) -> list[dict]:
        """List all Zotero collections."""
        collections = self._call_zotero(
            "list_collections",
            self.zot.collections,
        )
        result = []
        for c in collections:
            data = c.get("data", {})
            meta = c.get("meta", {})
            result.append({
                "key": data.get("key", ""),
                "name": data.get("name", ""),
                "parent_key": data.get("parentCollection", ""),
                "num_items": meta.get("numItems", 0),
            })
        return result

    def find_collection_by_name(self, name: str, parent_key: str | None = None) -> str | None:
        """Find a collection by name (case-insensitive), optionally under a parent.

        Args:
            name: Collection name to find.
            parent_key: If specified, only match collections with this parentCollection.

        Returns:
            Collection key or None.
        """
        collections = self.list_collections()
        name_lower = name.lower()
        for c in collections:
            if c["name"].lower() == name_lower:
                if parent_key is not None:
                    if c.get("parent_key", "") == parent_key:
                        return c["key"]
                else:
                    return c["key"]
        return None

    def create_collection(self, name: str, parent_key: str | None = None) -> str:
        """Create a new collection, optionally as a subcollection. Returns its key."""
        payload = {"name": name}
        if parent_key:
            payload["parentCollection"] = parent_key

        resp = self._call_zotero(
            "create_collection",
            self.zot.create_collections,
            [payload],
        )

        if not resp or "successful" not in resp:
            raise ZoteroError(f"Unexpected response from create_collections: {resp}")

        successful = resp.get("successful", {})
        if not successful:
            failed = resp.get("failed", {})
            raise ZoteroError(f"Failed to create collection: {failed}")

        key = successful["0"]["key"]
        parent_info = f" (parent: {parent_key})" if parent_key else ""
        logger.info("Created Zotero collection: %s (%s)%s", key, name, parent_info)
        return key

    def find_or_create_collection(self, name: str, parent_key: str | None = None) -> str:
        """Find a collection by name, or create it. Returns key.

        Args:
            name: Collection name.
            parent_key: If specified, find/create under this parent collection.
        """
        key = self.find_collection_by_name(name, parent_key=parent_key)
        if key:
            logger.info("Found existing collection '%s' -> %s", name, key)
            return key
        logger.info("Collection '%s' not found, creating...", name)
        return self.create_collection(name, parent_key=parent_key)

    # ─── Items without PDFs ──────────────────────────────────────

    def get_items_without_pdfs(self, collection_key: str, limit: int = 50) -> list[dict]:
        """Get items in a collection that have no PDF attachments.

        Uses collection_items_top() for top-level items, then checks
        children() for each to see if any child has itemType='attachment'
        with contentType='application/pdf'.
        """
        items = self._call_zotero(
            "get_collection_items_top",
            self.zot.collection_items_top,
            collection_key,
            limit=limit,
        )

        no_pdf = []
        for item in items:
            data = item.get("data", {})
            meta = item.get("meta", {})
            # Skip non-regular items (notes, attachments themselves)
            if data.get("itemType") in ("attachment", "note"):
                continue
            # Quick check: if numChildren == 0, definitely no PDF
            if meta.get("numChildren", 0) == 0:
                no_pdf.append(self._format_item(item))
                continue
            # Has children — check if any are PDF attachments
            children = self._call_zotero(
                "get_children",
                self.zot.children,
                data["key"],
            )
            has_pdf = any(
                c.get("data", {}).get("itemType") == "attachment"
                and "pdf" in c.get("data", {}).get("contentType", "").lower()
                for c in children
            )
            if not has_pdf:
                no_pdf.append(self._format_item(item))

        logger.info(
            "get_items_without_pdfs: %d/%d items lack PDF attachments in collection %s",
            len(no_pdf), len(items), collection_key,
        )
        return no_pdf

    # ─── PDF retrieval from Zotero ────────────────────────────────

    def get_pdf_attachment(self, item_key: str) -> dict | None:
        """Find a PDF attachment for a Zotero item.

        Args:
            item_key: The Zotero item key.

        Returns:
            Attachment metadata dict or None if no PDF found.
        """
        children = self._call_zotero(
            "get_children",
            self.zot.children,
            item_key,
        )
        for child in children:
            data = child.get("data", {})
            if (
                data.get("itemType") == "attachment"
                and data.get("contentType", "").lower() == "application/pdf"
            ):
                return data
        return None

    def download_pdf(self, item_key: str, dest_dir: str) -> dict:
        """Download a PDF from Zotero for a given item.

        Finds the PDF attachment, downloads it via the Zotero API,
        and writes it to dest_dir.

        Args:
            item_key: The Zotero item key.
            dest_dir: Directory to write the PDF file.

        Returns:
            dict with keys: success, path, size_bytes, error.
        """
        try:
            attachment = self.get_pdf_attachment(item_key)
            if not attachment:
                return {
                    "success": False,
                    "path": None,
                    "size_bytes": 0,
                    "error": "No PDF attachment found for this item.",
                }

            attachment_key = attachment["key"]
            file_content = self._call_zotero(
                "download_file",
                self.zot.file,
                attachment_key,
            )

            os.makedirs(dest_dir, exist_ok=True)
            pdf_path = os.path.join(dest_dir, f"{attachment_key}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(file_content)

            size_bytes = len(file_content)
            logger.info(
                "Downloaded PDF from Zotero: %s (%d bytes)",
                pdf_path, size_bytes,
            )
            return {
                "success": True,
                "path": pdf_path,
                "size_bytes": size_bytes,
                "error": None,
            }
        except ZoteroError as e:
            return {
                "success": False,
                "path": None,
                "size_bytes": 0,
                "error": str(e),
            }

    def find_item_by_record_id(self, record_id: str) -> dict | None:
        """Find a Zotero item by its SFU Library record ID.

        Searches the Zotero library for items whose 'extra' field
        contains 'SFU-Library-RecordID: {record_id}'.

        Args:
            record_id: The SFU Library record ID.

        Returns:
            Formatted item dict or None if not found.
        """
        items = self._call_zotero(
            "find_by_record_id",
            self.zot.items,
            q=record_id,
            limit=5,
        )
        for item in items:
            data = item.get("data", {})
            extra = data.get("extra", "")
            if f"SFU-Library-RecordID: {record_id}" in extra:
                return self._format_item(item)
        return None

    # ─── Search and browsing ───────────────────────────────────────

    def search_items(self, query: str, limit: int = 20) -> list[dict]:
        """Search the user's Zotero library. Returns formatted item list."""
        items = self._call_zotero(
            "search_items",
            self.zot.items,
            q=query,
            limit=limit,
            sort="date",
            direction="desc",
        )
        logger.info("Zotero search for '%s': %d results", query, len(items))
        return [self._format_item(item) for item in items]

    def get_collection_items(self, collection_key: str, limit: int = 50) -> list[dict]:
        """Get items in a specific collection. Returns formatted item list."""
        items = self._call_zotero(
            "get_collection_items",
            self.zot.collection_items,
            collection_key,
            limit=limit,
        )
        return [self._format_item(item) for item in items]

    def _format_item(self, item: dict) -> dict:
        """Format a raw pyzotero item dict into a clean dict."""
        data = item.get("data", {})
        creators = data.get("creators", [])
        author_names = []
        for c in creators:
            last = c.get("lastName", "")
            first = c.get("firstName", "")
            if last and first:
                author_names.append(f"{last}, {first}")
            elif last:
                author_names.append(last)
            elif c.get("name"):
                author_names.append(c["name"])

        collections = data.get("collections", [])
        tags = [t.get("tag", "") for t in data.get("tags", [])]

        return {
            "key": data.get("key", ""),
            "title": data.get("title", ""),
            "authors": author_names,
            "date": data.get("date", ""),
            "item_type": data.get("itemType", ""),
            "DOI": data.get("DOI", ""),
            "ISBN": data.get("ISBN", ""),
            "publication": data.get("publicationTitle", "") or data.get("publisher", ""),
            "collections": collections,
            "tags": tags,
        }

    def format_item_summary(self, item: dict) -> str:
        """Format a Zotero item dict into a human-readable string."""
        parts = []
        title = item.get("title", "Untitled")
        parts.append(f"Title: {title}")

        authors = item.get("authors", [])
        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")

        date = item.get("date", "")
        if date:
            parts.append(f"Date: {date}")

        item_type = item.get("item_type", "")
        if item_type:
            parts.append(f"Type: {item_type}")

        pub = item.get("publication", "")
        if pub:
            parts.append(f"Publication: {pub}")

        doi = item.get("DOI", "")
        if doi:
            parts.append(f"DOI: {doi}")

        isbn = item.get("ISBN", "")
        if isbn:
            parts.append(f"ISBN: {isbn}")

        tags = item.get("tags", [])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        collections = item.get("collections", [])
        if collections:
            parts.append(f"Collections: {', '.join(collections)}")

        key = item.get("key", "")
        if key:
            parts.append(f"Zotero Key: {key}")

        return "\n".join(parts)

    # ─── Duplicate detection ───────────────────────────────────────

    def check_duplicate(self, metadata: dict) -> dict:
        """Multi-signal deconfliction before saving.

        Checks in priority order:
        1. DOI match (exact)
        2. ISBN match (exact)
        3. Title + author match (fuzzy)

        Returns dict with: is_duplicate, existing_key, match_type,
        existing_item_summary.
        """
        result = {
            "is_duplicate": False,
            "existing_key": None,
            "match_type": None,
            "existing_item_summary": None,
        }

        # 1. DOI match
        doi = metadata.get("doi", "")
        if doi:
            try:
                items = self._call_zotero("check_dup_doi", self.zot.items, q=doi, limit=5)
                for item in items:
                    data = item.get("data", {})
                    if data.get("DOI", "").strip().lower() == doi.strip().lower():
                        formatted = self._format_item(item)
                        result["is_duplicate"] = True
                        result["existing_key"] = data.get("key", "")
                        result["match_type"] = "DOI"
                        result["existing_item_summary"] = self.format_item_summary(formatted)
                        return result
            except ZoteroError:
                logger.warning("DOI duplicate check failed, continuing")

        # 2. ISBN match
        isbn = metadata.get("isbn", "")
        if isbn:
            try:
                items = self._call_zotero("check_dup_isbn", self.zot.items, q=isbn, limit=5)
                for item in items:
                    data = item.get("data", {})
                    if data.get("ISBN", "").replace("-", "") == isbn.replace("-", ""):
                        formatted = self._format_item(item)
                        result["is_duplicate"] = True
                        result["existing_key"] = data.get("key", "")
                        result["match_type"] = "ISBN"
                        result["existing_item_summary"] = self.format_item_summary(formatted)
                        return result
            except ZoteroError:
                logger.warning("ISBN duplicate check failed, continuing")

        # 3. Title + author fuzzy match
        title = metadata.get("title", "")
        if title:
            try:
                items = self._call_zotero("check_dup_title", self.zot.items, q=title[:100], limit=10)
                authors = metadata.get("authors") or metadata.get("creators", [])
                first_author_last = self._extract_last_name(authors[0] if authors else "")

                for item in items:
                    data = item.get("data", {})
                    existing_title = data.get("title", "")
                    similarity = self._title_similarity(title, existing_title)

                    existing_creators = data.get("creators", [])
                    existing_first_last = ""
                    if existing_creators:
                        existing_first_last = existing_creators[0].get("lastName", "")

                    author_match = (
                        first_author_last.lower() == existing_first_last.lower()
                        if first_author_last and existing_first_last
                        else False
                    )

                    if similarity > 0.85 and author_match:
                        formatted = self._format_item(item)
                        result["is_duplicate"] = True
                        result["existing_key"] = data.get("key", "")
                        result["match_type"] = "title+author"
                        result["existing_item_summary"] = self.format_item_summary(formatted)
                        return result
            except ZoteroError:
                logger.warning("Title+author duplicate check failed, continuing")

        return result

    @staticmethod
    def _extract_last_name(author_str: str) -> str:
        """Extract last name from an author string."""
        if not author_str:
            return ""
        # Strip Ex Libris suffix
        if "$$" in author_str:
            author_str = author_str.split("$$")[0].strip()
        if "," in author_str:
            return author_str.split(",")[0].strip()
        parts = author_str.strip().split()
        return parts[-1] if parts else ""

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """Compute normalized title similarity (case-insensitive, stripped punctuation)."""
        def _normalize(s: str) -> str:
            s = s.lower()
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        na, nb = _normalize(a), _normalize(b)
        if not na or not nb:
            return 0.0
        if na == nb:
            return 1.0

        # Use token overlap (Jaccard similarity)
        tokens_a = set(na.split())
        tokens_b = set(nb.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
