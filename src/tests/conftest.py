"""Shared test fixtures for SFU Library MCP server tests."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def sample_pnx_record():
    """Realistic PNX JSON record for citation/formatting tests."""
    return {
        "pnx": {
            "display": {
                "title": ["Machine Learning: A Probabilistic Perspective"],
                "creator": [
                    "Murphy, Kevin P.$$QMurphy, Kevin P."
                ],
                "contributor": [],
                "creationdate": ["2012"],
                "publisher": ["MIT Press"],
                "type": ["book"],
                "source": [""],
                "description": [
                    "A comprehensive introduction to machine learning that uses probabilistic models."
                ],
                "subject": [
                    "Machine learning",
                    "Probabilities",
                    "Artificial intelligence"
                ],
            },
            "addata": {
                "isbn": ["9780262018029"],
                "issn": [],
                "doi": [],
                "volume": [],
                "issue": [],
                "spage": [],
                "epage": [],
                "pages": [],
            },
            "control": {
                "recordid": ["alma991234567890"]
            },
            "links": {
                "linktohtml": [],
                "linktopdf": [],
                "linktorsrc": ["https://example.com/fulltext"],
                "openaccess": [],
            },
            "delivery": {
                "availability": ["available"],
                "holding": [
                    {
                        "libraryCode": "SFUL",
                        "subLocationCode": "STACKS",
                        "callNumber": "Q325.5 .M87 2012",
                    }
                ],
            },
        }
    }


@pytest.fixture
def sample_article_record():
    """Realistic PNX JSON record for a journal article."""
    return {
        "pnx": {
            "display": {
                "title": ["Deep Learning for Natural Language Processing"],
                "creator": [
                    "Smith, John A.$$QSmith, John A.",
                    "Doe, Jane B.$$QDoe, Jane B."
                ],
                "contributor": [],
                "creationdate": ["2023"],
                "publisher": [""],
                "type": ["article"],
                "source": ["Journal of Artificial Intelligence Research"],
                "description": [
                    "A survey of deep learning techniques for NLP tasks."
                ],
                "subject": ["Deep learning", "Natural language processing"],
            },
            "addata": {
                "isbn": [],
                "issn": ["1076-9757"],
                "doi": ["10.1613/jair.1.12345"],
                "volume": ["76"],
                "issue": ["3"],
                "spage": ["1"],
                "epage": ["45"],
                "pages": ["1-45"],
            },
            "control": {
                "recordid": ["alma999888777666"]
            },
            "links": {
                "linktohtml": ["https://example.com/article.html"],
                "linktopdf": ["https://example.com/article.pdf"],
                "linktorsrc": [],
                "openaccess": ["https://example.com/oa"],
            },
            "delivery": {
                "availability": ["available_online"],
            },
        }
    }


@pytest.fixture
def mock_search_response(sample_pnx_record, sample_article_record):
    """Realistic search API response."""
    return {
        "docs": [sample_pnx_record, sample_article_record],
        "info": {
            "total": 2,
            "first": 0,
            "last": 1,
        },
    }


@pytest.fixture
def mock_config():
    """ServerConfig with test values."""
    from lib.config import ServerConfig

    return ServerConfig(
        search_timeout=10,
        max_retries=2,
        log_level="DEBUG",
        features={"cache_enabled": True, "retry_enabled": True},
    )
