from flask import Flask, jsonify
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

from .config import Config


db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "请先登录后再访问后台。"
login_manager.login_message_category = "warning"


def create_app(config_class=Config):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized():
        if request_wants_json():
            return jsonify({"error": "authentication_required"}), 401
        from flask import redirect, request, url_for

        return redirect(url_for("auth.login", next=request.full_path))

    from .auth.routes import auth_bp
    from .admin.routes import admin_bp
    from .api.routes import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    register_cli(app)
    register_error_handlers(app)

    return app


def request_wants_json():
    from flask import request

    return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"


def register_cli(app):
    @app.cli.command("init-db")
    def init_db_command():
        from .tui import initialize_database

        print(initialize_database(app))

    @app.cli.command("tui")
    def tui_command():
        from .tui import run_tui

        run_tui(app)


def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(error):
        if request_wants_json():
            return jsonify({"error": "not_found"}), 404
        return error

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        if request_wants_json():
            return jsonify({"error": "internal_server_error"}), 500
        return error
