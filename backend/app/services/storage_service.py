"""Supabase Storage operations."""

from __future__ import annotations

from typing import Protocol, cast

from app.clients.supabase_client import SupabaseClientProtocol
from app.core.exceptions import ConfigurationError


class StorageBucketProtocol(Protocol):
    def upload(
        self, path: str, file: bytes, file_options: dict[str, object] | None = None
    ) -> object: ...
    def download(self, path: str) -> bytes: ...
    def remove(self, paths: list[str]) -> object: ...


class StorageProtocol(Protocol):
    def from_(self, bucket: str) -> StorageBucketProtocol: ...


class ClientWithStorageProtocol(SupabaseClientProtocol, Protocol):
    storage: StorageProtocol


class StorageService:
    def __init__(self, client: SupabaseClientProtocol, bucket: str) -> None:
        self.client = cast(ClientWithStorageProtocol, client)
        self.bucket = bucket
        if not bucket:
            raise ConfigurationError("SUPABASE_STORAGE_BUCKET is not configured.")

    def upload_csv(self, path: str, content: bytes) -> None:
        self.client.storage.from_(self.bucket).upload(
            path,
            content,
            {"content-type": "text/csv", "upsert": "false"},
        )

    def download(self, path: str) -> bytes:
        return self.client.storage.from_(self.bucket).download(path)

    def delete(self, path: str) -> None:
        self.client.storage.from_(self.bucket).remove([path])
