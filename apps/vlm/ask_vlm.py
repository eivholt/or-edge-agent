import base64
import mimetypes
import os
from pathlib import Path

import anthropic
import logfire
from dotenv import load_dotenv

load_dotenv(override=True)

logfire.configure(service_name="or-edge-agent")
logfire.instrument_anthropic()

AZURE_ANTHROPIC_ENDPOINT = os.getenv(
    "AZURE_ANTHROPIC_ENDPOINT",
    "https://synthetic-patient-resource.services.ai.azure.com/anthropic/",
)
AZURE_VLM_DEPLOYMENT = os.getenv("AZURE_VLM_DEPLOYMENT", "claude-opus-4-7")
AZURE_VLM_API_KEY = os.getenv("AZURE_VLM_API_KEY")

VLM_SYSTEM_PROMPT = (
    "You are a surgical-scene analyst. Answer the user's question about the "
    "image. Respond ONLY with a JSON object — no markdown, no commentary:\n"
    '{"answer": true/false, "description": "one or two sentences"}\n'
    '"answer" is true when the answer to the question is YES, false when NO.'
)


@logfire.instrument("ask_vlm question={question}")
def ask_vlm(image_path: str, question: str) -> str:
    client = anthropic.Anthropic(
        base_url=AZURE_ANTHROPIC_ENDPOINT,
        api_key=AZURE_VLM_API_KEY,
    )

    p = Path(image_path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")

    r = client.messages.create(
        model=AZURE_VLM_DEPLOYMENT,
        max_tokens=300,
        system=VLM_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": question},
            ],
        }],
    )

    return r.content[0].text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("question")
    args = parser.parse_args()

    print(ask_vlm(args.image, args.question))