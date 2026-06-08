"""
API Integration Tests
========================
Tests for the FastAPI backend endpoints.
"""

import pytest


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, api_client):
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_register(self, api_client):
        response = api_client.post("/api/v1/auth/register", json={
            "email": "test@example.com",
            "password": "testpass123",
            "full_name": "Test User",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["role"] == "user"

    def test_register_duplicate_email(self, api_client):
        # Register first
        api_client.post("/api/v1/auth/register", json={
            "email": "dup@example.com",
            "password": "testpass123",
        })
        # Try again
        response = api_client.post("/api/v1/auth/register", json={
            "email": "dup@example.com",
            "password": "testpass123",
        })
        assert response.status_code == 400

    def test_login(self, api_client):
        # Register
        api_client.post("/api/v1/auth/register", json={
            "email": "login@example.com",
            "password": "testpass123",
        })
        # Login
        response = api_client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": "testpass123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, api_client):
        api_client.post("/api/v1/auth/register", json={
            "email": "wrong@example.com",
            "password": "correct123",
        })
        response = api_client.post("/api/v1/auth/login", json={
            "email": "wrong@example.com",
            "password": "incorrect123",
        })
        assert response.status_code == 401

    def test_get_me_authenticated(self, api_client):
        # Register + Login
        api_client.post("/api/v1/auth/register", json={
            "email": "me@example.com",
            "password": "testpass123",
            "full_name": "Me User",
        })
        login = api_client.post("/api/v1/auth/login", json={
            "email": "me@example.com",
            "password": "testpass123",
        })
        token = login.json()["access_token"]

        response = api_client.get("/api/v1/auth/me",
                                  headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["email"] == "me@example.com"


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""

    def test_analytics_summary(self, api_client):
        response = api_client.get("/api/v1/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_images_processed" in data
        assert "total_detections" in data

    def test_analytics_timeline(self, api_client):
        response = api_client.get("/api/v1/analytics/timeline?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "period" in data

    def test_analytics_classes(self, api_client):
        response = api_client.get("/api/v1/analytics/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data

    def test_analytics_recent(self, api_client):
        response = api_client.get("/api/v1/analytics/recent?limit=5")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestUploadEndpoints:
    """Tests for upload endpoints."""

    def test_upload_invalid_type(self, api_client):
        """Uploading a non-image file should fail."""
        response = api_client.post(
            "/api/v1/upload/image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


class TestPredictEndpoints:
    """Tests for prediction list endpoint."""

    def test_list_detections_empty(self, api_client):
        response = api_client.get("/api/v1/predict/")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_get_nonexistent_detection(self, api_client):
        response = api_client.get("/api/v1/predict/999")
        assert response.status_code == 404
