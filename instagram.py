import os

import requests


INSTAGRAM_ACCESS_TOKEN = os.getenv(
    "INSTAGRAM_ACCESS_TOKEN"
)

INSTAGRAM_API_VERSION = os.getenv(
    "INSTAGRAM_API_VERSION",
    "v23.0"
)


def send_instagram_message(
    recipient_id: str,
    message: str
):

    url = (
        f"https://graph.instagram.com/"
        f"{INSTAGRAM_API_VERSION}/me/messages"
    )

    payload = {
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message
        }
    }

    headers = {
        "Authorization": (
            f"Bearer {INSTAGRAM_ACCESS_TOKEN}"
        ),
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return response.json()