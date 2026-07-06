"""
config.py
---------
Centralized Flask configuration. Values are read from environment
variables where possible so the same codebase can run in development
and production (e.g. on Render/Heroku) without code changes -- only
environment variables differ.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Base configuration shared by all environments."""

    # SECRET_KEY is used by Flask to sign session cookies. In production
    # this MUST be overridden via the SECRET_KEY environment variable.
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # SQLite database lives in the instance/ folder, which Flask keeps
    # outside version control by convention (see .gitignore).
    DATABASE_PATH = os.path.join(BASE_DIR, "instance", "student_performance.db")

    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")

    # Maximum number of recent predictions returned by /api/history
    HISTORY_LIMIT = 100

    # JSON responses keep key order for a more readable payload
    JSON_SORT_KEYS = False


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


def get_config():
    env = os.environ.get("FLASK_ENV", "development")
    return ProductionConfig if env == "production" else DevelopmentConfig
