# Flask 后台管理系统

这是一个基于 Flask 的简洁后台管理系统，包含用户认证、后台页面、REST API 和 SQLite 数据库模型。

## 功能

- Flask app factory：`app.create_app`
- 数据库：Flask-SQLAlchemy，默认使用 `admin.sqlite3`
- 用户认证：Flask-Login 登录、登出、后台页面保护
- 后台页面：仪表盘、用户列表
- REST API：健康检查、当前用户、用户列表
- CLI 初始化数据库并创建默认管理员
- pytest 自动化测试

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

如果 Windows 环境中 `python` 不可用，可以尝试：

```bash
py -m pip install -r requirements.txt
```

## 初始化数据库

```bash
python -m flask --app run.py init-db
```

默认会创建管理员账号：

- 用户名：`admin`
- 密码：`admin123`

## 启动项目

```bash
python run.py
```

或使用 Flask CLI：

```bash
python -m flask --app run.py run --debug
```

启动后访问：

- 登录页：`http://127.0.0.1:5000/login`
- 后台首页：`http://127.0.0.1:5000/admin`
- 用户管理：`http://127.0.0.1:5000/admin/users`

## API 示例

### 健康检查

```bash
curl http://127.0.0.1:5000/api/health
```

响应示例：

```json
{"status":"ok"}
```

### 当前用户

需要先通过页面登录，或在客户端中携带登录后的 session cookie。

```bash
curl http://127.0.0.1:5000/api/me
```

未登录响应：

```json
{"error":"authentication_required"}
```

### 用户列表

仅管理员可访问。

```bash
curl http://127.0.0.1:5000/api/users
```

非管理员响应：

```json
{"error":"admin_required"}
```

## 运行测试

```bash
python -m pytest
```

## 查看路由

```bash
python -m flask --app run.py routes
```

## 配置

可通过环境变量覆盖默认配置：

- `SECRET_KEY`：Flask session 密钥
- `DATABASE_URL`：数据库连接地址，默认是项目根目录下的 SQLite 文件 `admin.sqlite3`
