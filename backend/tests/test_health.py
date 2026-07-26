"""Health and secret-safety tests."""


def test_health_works_without_database_or_model(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database_configured": False,
        "storage_configured": False,
        "model_ready": False,
    }


def test_versioned_health_and_openapi_are_available(client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert "/api/v1/predictions" in client.get("/openapi.json").json()["paths"]


def test_health_never_exposes_secrets(client) -> None:
    body = client.get("/health").text.lower()
    assert "service_role" not in body
    assert "anon_key" not in body
    assert "supabase_url" not in body
