from flask import Blueprint, redirect, render_template, url_for
from flask_login import login_required

from app.models import User


admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/")
def index():
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/admin")
@login_required
def dashboard():
    user_count = User.query.count()
    admin_count = User.query.filter_by(is_admin=True).count()
    latest_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    return render_template(
        "admin/dashboard.html",
        user_count=user_count,
        admin_count=admin_count,
        latest_users=latest_users,
    )


@admin_bp.route("/admin/users")
@login_required
def users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin/users.html", users=users)
