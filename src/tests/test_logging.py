"""Tests for logging_setup module."""

import logging
import os
import sys

from logging.handlers import RotatingFileHandler

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
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
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


class TestFileHandler:
    def test_file_handler_created_when_log_file_set(self, tmp_path):
        """A RotatingFileHandler should be added when log_file is provided."""
        log_path = str(tmp_path / "test.log")
        logger = setup_logging(name="test_file_handler", log_file=log_path)
        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == log_path

    def test_file_handler_writes_debug(self, tmp_path):
        """File handler should capture DEBUG messages even if stderr is INFO."""
        log_path = str(tmp_path / "test_debug.log")
        logger = setup_logging(level="INFO", name="test_file_debug", log_file=log_path)
        logger.debug("This is a debug message for file only")
        logger.info("This is an info message")

        # Flush handlers
        for h in logger.handlers:
            h.flush()

        content = open(log_path).read()
        assert "debug message for file only" in content
        assert "info message" in content

    def test_file_handler_always_debug_level(self, tmp_path):
        """File handler level should always be DEBUG regardless of configured level."""
        log_path = str(tmp_path / "test_level.log")
        logger = setup_logging(level="WARNING", name="test_file_level", log_file=log_path)
        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers[0].level == logging.DEBUG

    def test_no_file_handler_when_empty_string(self):
        """No file handler should be added when log_file is empty."""
        logger = setup_logging(name="test_no_file", log_file="")
        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 0

    def test_file_handler_graceful_on_bad_path(self, tmp_path):
        """Bad log file path should not crash, just warn."""
        logger = setup_logging(name="test_bad_path", log_file="/nonexistent/dir/test.log")
        # Should still have stderr handler at minimum
        assert len(logger.handlers) >= 1

    def test_stderr_handler_still_present_with_file(self, tmp_path):
        """Both stderr and file handlers should exist when log_file is set."""
        log_path = str(tmp_path / "test_both.log")
        logger = setup_logging(name="test_both_handlers", log_file=log_path)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types or any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
            for h in logger.handlers
        )
        assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
