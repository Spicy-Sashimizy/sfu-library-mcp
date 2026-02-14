"""Tests for logging_setup module."""

import logging
import sys

from lib.logging_setup import setup_logging


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging(name="test_returns")
        assert isinstance(logger, logging.Logger)

    def test_writes_to_stderr_not_stdout(self):
        logger = setup_logging(name="test_stderr")
        handlers = logger.handlers
        assert len(handlers) >= 1
        for handler in handlers:
            if isinstance(handler, logging.StreamHandler):
                assert handler.stream is sys.stderr

    def test_level_configuration(self):
        logger = setup_logging(level="DEBUG", name="test_level_debug")
        assert logger.level == logging.DEBUG

        logger2 = setup_logging(level="WARNING", name="test_level_warn")
        assert logger2.level == logging.WARNING

    def test_no_duplicate_handlers(self):
        name = "test_no_dup"
        logger1 = setup_logging(name=name)
        count1 = len(logger1.handlers)
        logger2 = setup_logging(name=name)
        count2 = len(logger2.handlers)
        assert count1 == count2
        assert logger1 is logger2

    def test_invalid_level_defaults_to_info(self):
        logger = setup_logging(level="NOTAVALIDLEVEL", name="test_invalid_level")
        assert logger.level == logging.INFO
