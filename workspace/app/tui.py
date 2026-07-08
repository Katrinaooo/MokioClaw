MENU_TEXT = """\nFlask 后台管理 TUI\n1. 查看系统状态\n2. 查看用户列表\n3. 初始化数据库\nq. 退出\n""".strip()


def get_status_lines(app):
    from . import db
    from .models import User

    with app.app_context():
        db.create_all()
        user_count = User.query.count()
        admin_count = User.query.filter_by(is_admin=True).count()
        return [
            "系统状态",
            f"数据库：{app.config['SQLALCHEMY_DATABASE_URI']}",
            f"用户数量：{user_count}",
            f"管理员数量：{admin_count}",
        ]


def get_user_lines(app):
    from . import db
    from .models import User

    with app.app_context():
        db.create_all()
        users = User.query.order_by(User.id.asc()).all()
        if not users:
            return ["用户列表", "暂无用户"]
        return ["用户列表", *[format_user(user) for user in users]]


def initialize_database(app):
    from . import db
    from .models import User

    with app.app_context():
        db.create_all()
        admin = User.query.filter_by(username="admin").first()
        if admin is not None:
            return "数据库已初始化，默认管理员已存在。"

        admin = User(username="admin", email="admin@example.com", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        return "数据库已初始化，默认管理员：admin / admin123"


def format_user(user):
    role = "管理员" if user.is_admin else "普通用户"
    return f"#{user.id} {user.username} <{user.email}> - {role}"


def print_lines(lines, output_func):
    for line in lines:
        output_func(line)


def run_tui(app, input_func=input, output_func=print):
    while True:
        output_func(MENU_TEXT)
        choice = input_func("请选择操作：").strip().lower()

        if choice in {"q", "quit", "exit"}:
            output_func("已退出。")
            return
        if choice == "1":
            print_lines(get_status_lines(app), output_func)
        elif choice == "2":
            print_lines(get_user_lines(app), output_func)
        elif choice == "3":
            output_func(initialize_database(app))
        else:
            output_func("无效选项，请重新输入。")
