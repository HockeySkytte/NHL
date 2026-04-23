import os
from datetime import timedelta

from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY') or os.getenv('SECRET_KEY') or 'nhl-dev-secret-change-me'
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    from .routes import main_bp, preload_common_models, start_prestart_logger
    app.register_blueprint(main_bp)

    # Eager-load common xG models at startup to reduce first-request latency (toggle via XG_PRELOAD=1)
    try:
        if os.getenv('XG_PRELOAD', '1') == '1':
            preload_common_models()
    except Exception:
        # Never block app startup on preload issues
        pass

    # Start background prestart snapshot logger (toggle via PRESTART_LOGGER=0)
    try:
        if os.getenv('PRESTART_LOGGER', '1') == '1':
            start_prestart_logger()
    except Exception:
        # Never block app startup on background thread issues
        pass

    return app
