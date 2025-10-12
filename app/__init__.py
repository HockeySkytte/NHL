from flask import Flask


def create_app():
    app = Flask(__name__)

    from .routes import main_bp, preload_common_models
    app.register_blueprint(main_bp)

    # Eager-load common xG models at startup to reduce first-request latency (toggle via XG_PRELOAD=1)
    import os
    try:
        if os.getenv('XG_PRELOAD', '1') == '1':
            preload_common_models()
    except Exception:
        # Never block app startup on preload issues
        pass

    return app
