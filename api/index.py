import os
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from legal_ai import get_legal_answer

from instagram import send_instagram_message


app = FastAPI()


VERIFY_TOKEN = os.getenv(
    "INSTAGRAM_VERIFY_TOKEN"
)


conversations: Dict[
    str,
    List
] = {}


@app.get("/api/webhook")
async def verify_webhook(
    request: Request
):

    params = request.query_params

    mode = params.get(
        "hub.mode"
    )

    token = params.get(
        "hub.verify_token"
    )

    challenge = params.get(
        "hub.challenge"
    )

    if (
        mode == "subscribe"
        and token == VERIFY_TOKEN
    ):

        return PlainTextResponse(
            challenge
        )

    return PlainTextResponse(
        "Verification failed",
        status_code=403
    )


@app.post("/api/webhook")
async def receive_webhook(
    request: Request
):

    body = await request.json()

    try:

        for entry in body.get(
            "entry",
            []
        ):

            for messaging_event in entry.get(
                "messaging",
                []
            ):

                sender = messaging_event.get(
                    "sender",
                    {}
                )

                sender_id = sender.get(
                    "id"
                )

                message = messaging_event.get(
                    "message",
                    {}
                )

                text = message.get(
                    "text"
                )

                if not sender_id or not text:
                    continue

                if sender_id not in conversations:

                    conversations[
                        sender_id
                    ] = []

                conversations[
                    sender_id
                ].append(
                    HumanMessage(
                        content=text
                    )
                )

                try:

                    answer = get_legal_answer(
                        conversations[
                            sender_id
                        ]
                    )

                    conversations[
                        sender_id
                    ].append(
                        AIMessage(
                            content=answer
                        )
                    )

                    send_instagram_message(
                        sender_id,
                        answer
                    )

                except Exception as e:

                    print(
                        f"AI error: {e}"
                    )

        return {
            "status": "ok"
        }

    except Exception as e:

        print(
            f"Webhook error: {e}"
        )

        return {
            "status": "error"
        }