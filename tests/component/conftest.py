"""Component test fixtures.

All module-level mocking is handled by the root tests/conftest.py.
This file provides component-test-specific fixtures.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_redis_async():
    """Provide a fresh AsyncMock for Redis async client."""
    client = AsyncMock()
    client.get.return_value = None
    client.brpop.return_value = None
    return client


@pytest.fixture
def mock_redis_sync():
    """Provide a fresh Mock for Redis sync client."""
    return MagicMock()
