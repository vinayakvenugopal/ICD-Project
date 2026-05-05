import os

import django


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ICD.settings")
django.setup()

from django.contrib.auth import get_user_model  # noqa: E402


email = os.environ.get("DJANGO_SUPERUSER_EMAIL")
password = os.environ.get("DJANGO_SUPERUSER_PASSWORD")

if not email or not password:
    print("Skipping superuser creation; DJANGO_SUPERUSER_EMAIL or DJANGO_SUPERUSER_PASSWORD is not set.")
else:
    User = get_user_model()
    user, created = User.objects.get_or_create(
        email=email,
        defaults={
            "fullname": os.environ.get("DJANGO_SUPERUSER_NAME", "Admin"),
            "role": "admin",
            "status": "accepted",
            "is_staff": True,
            "is_superuser": True,
        },
    )
    if created:
        user.set_password(password)
        user.save()
        print(f"Created superuser {email}.")
    else:
        updated = False
        if not user.is_staff or not user.is_superuser:
            user.is_staff = True
            user.is_superuser = True
            updated = True
        if os.environ.get("DJANGO_RESET_SUPERUSER_PASSWORD", "False").lower() == "true":
            user.set_password(password)
            updated = True
        if updated:
            user.save()
        print(f"Superuser {email} already exists.")
