"""Reusable Supabase client provider."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError


class SupabaseClientProtocol(Protocol):
    """Minimal boundary used by repositories and services."""

    def table(self, table_name: str) -> object: ...


@lru_cache(maxsize=1)
def get_supabase_client() -> SupabaseClientProtocol:
    """Create one process-wide service-role client lazily."""
    settings: Settings = get_settings()
    if not settings.supabase_configured:
        raise ConfigurationError(
            "Supabase is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY."
        )
    try:
        from supabase import create_client
    except ImportError as exc:
        raise ConfigurationError(
            "The supabase package is not installed."
        ) from exc
    service_key = settings.supabase_service_role_key
    if service_key is None or settings.supabase_url is None:
        raise ConfigurationError()
    return create_client(settings.supabase_url, service_key.get_secret_value())


def clear_supabase_client_cache() -> None:
    """Support deterministic configuration in tests."""
    get_supabase_client.cache_clear()
