"""Supabase bearer-token verification isolated from route logic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast
from uuid import UUID

from app.clients.supabase_client import SupabaseClientProtocol
from app.core.exceptions import AuthenticationError
from app.repositories.base import QueryBuilderProtocol, response_rows
from app.schemas.common import CurrentUser


class AuthUserProtocol(Protocol):
    id: str
    email: str | None


class AuthResponseProtocol(Protocol):
    user: AuthUserProtocol | None


class AuthProtocol(Protocol):
    def get_user(self, jwt: str) -> AuthResponseProtocol: ...


class ClientWithAuthProtocol(SupabaseClientProtocol, Protocol):
    auth: AuthProtocol


def resolve_current_user(
    client: SupabaseClientProtocol, access_token: str
) -> CurrentUser:
    """Verify a Supabase access token and load its organization profile."""
    if not access_token:
        raise AuthenticationError()
    auth_client = cast(ClientWithAuthProtocol, client)
    try:
        auth_response = auth_client.auth.get_user(access_token)
    except Exception as exc:
        raise AuthenticationError("The bearer token is invalid or expired.") from exc
    auth_user = auth_response.user
    if auth_user is None:
        raise AuthenticationError("The bearer token has no authenticated user.")
    query = cast(
        QueryBuilderProtocol, client.table("user_profiles")
    )
    rows = response_rows(
        query.select("id, organization_id, email, role")
        .eq("id", auth_user.id)
        .limit(1)
        .execute()
    )
    if not rows or not rows[0].get("organization_id"):
        raise AuthenticationError(
            "The authenticated user has no organization profile."
        )
    profile = rows[0]
    return CurrentUser(
        id=UUID(auth_user.id),
        organization_id=UUID(str(profile["organization_id"])),
        email=str(profile.get("email") or auth_user.email)
        if profile.get("email") or auth_user.email
        else None,
        role=str(profile.get("role") or "member"),
    )
