import pytest

from app import create_app, db
from app.models import User


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = False


@pytest.fixture
def app():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        admin = User(username="admin", email="admin@example.com", is_admin=True)
        admin.set_password("admin123")
        user = User(username="user", email="user@example.com", is_admin=False)
        user.set_password("user123")
        db.session.add_all([admin, user])
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, username="admin", password="admin123"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def test_health_check(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_login_success(client):
    response = login(client)

    assert response.status_code == 200
    assert "仪表盘" in response.get_data(as_text=True)
    assert "admin" in response.get_data(as_text=True)


def test_login_failure(client):
    response = login(client, password="wrong")

    assert response.status_code == 200
    assert "用户名或密码不正确" in response.get_data(as_text=True)


def test_admin_page_requires_login(client):
    response = client.get("/admin")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_api_me_requires_login_json_error(client):
    response = client.get("/api/me")

    assert response.status_code == 401
    assert response.get_json() == {"error": "authentication_required"}


def test_admin_can_list_users(client):
    login(client)
    response = client.get("/api/users")

    assert response.status_code == 200
    payload = response.get_json()
    assert [user["username"] for user in payload["users"]] == ["user", "admin"]


def test_non_admin_cannot_list_users(client):
    login(client, "user", "user123")
    response = client.get("/api/users")

    assert response.status_code == 403
    assert response.get_json() == {"error": "admin_required"}
