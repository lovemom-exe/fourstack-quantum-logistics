"""Shared FastAPI test configuration without Supabase."""

from __future__ import annotations

from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import Settings, get_settings
from app.main import create_app
from app.schemas.common import CurrentUser


USER_ID = UUID("11111111-1111-1111-1111-111111111111")
ORG_ID = UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def authenticated_user() -> CurrentUser:
    return CurrentUser(
        id=USER_ID,
        organization_id=ORG_ID,
        email="tester@example.com",
        role="admin",
    )


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        supabase_url=None,
        supabase_service_role_key=None,
        model_artifact_dir=tmp_path / "missing-model",
        allow_mock_predictions=True,
    )


@pytest.fixture
def app(test_settings: Settings, authenticated_user: CurrentUser):
    application = create_app(test_settings)
    application.dependency_overrides[get_settings] = lambda: test_settings
    application.dependency_overrides[get_current_user] = lambda: authenticated_user
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)
