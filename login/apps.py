from django.apps import AppConfig


class LoginConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'login'

    def ready(self):
        # Pre-warm the ICD model in the background so the first request
        # doesn't block. Runs in a daemon thread — won't delay server start.
        import threading
        import os

        def _warm_up():
            try:
                from ml.predict import _load_detector
                _load_detector()
                print("[ICD] Neural detector pre-loaded successfully.")
            except Exception as exc:
                print(f"[ICD] Pre-load failed (fallback will be used): {exc}")

        # Skip during manage.py commands (migrate, collectstatic, etc.)
        # but always run inside gunicorn workers on Render.
        import sys
        is_management_cmd = len(sys.argv) > 1 and sys.argv[1] in (
            'migrate', 'collectstatic', 'makemigrations', 'shell', 'test', 'createsuperuser'
        )
        if not is_management_cmd:
            t = threading.Thread(target=_warm_up, daemon=True)
            t.start()
