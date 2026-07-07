from functools import wraps

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError

from app import db
from app.models import User


api_bp = Blueprint("api", __name__)


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_admin:
            return jsonify({"error": "admin_required"}), 403
        return view(*args, **kwargs)

    return wrapped


@api_bp.route("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.route("/me")
@login_required
def me():
    return jsonify({"user": current_user.to_dict()})


@api_bp.route("/users", methods=["GET", "POST"])
@admin_required
def users():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        username = (payload.get("username") or "").strip()
        email = (payload.get("email") or "").strip()
        password = payload.get("password") or ""
        is_admin = bool(payload.get("is_admin", False))

        if not username or not email or not password:
            return jsonify({"error": "username_email_password_required"}), 400

        user = User(username=username, email=email, is_admin=is_admin)
        user.set_password(password)
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return jsonify({"error": "username_or_email_exists"}), 409

        return jsonify({"user": user.to_dict()}), 201

    all_users = User.query.order_by(User.created_at.desc(), User.id.desc()).all()
    return jsonify({"users": [user.to_dict() for user in all_users]})
