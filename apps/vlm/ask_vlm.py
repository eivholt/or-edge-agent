import base64
import mimetypes
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

logfire.configure(service_name="or-edge-agent")
logfire.instrument_openai()

AZURE_VLM_ENDPOINT = os.getenv(
    "AZURE_VLM_ENDPOINT",
    "https://synthetic-patient-resource.services.ai.azure.com/openai/v1/responses",
)
AZURE_VLM_DEPLOYMENT = os.getenv("AZURE_VLM_DEPLOYMENT", "gpt-5.6-sol")
AZURE_VLM_API_KEY = os.getenv("AZURE_VLM_API_KEY")

VLM_SYSTEM_PROMPT = (
    "You are a surgical-scene analyst. Answer the user's question about the "
    "image. Respond ONLY with a JSON object — no markdown, no commentary:\n"
    '{"answer": true/false, "description": "one or two sentences"}\n'
    '"answer" is true when the answer to the question is YES, false when NO.'
)


@logfire.instrument("ask_vlm question={question}")
def ask_vlm(image_path: str, question: str) -> str:
    base_url = AZURE_VLM_ENDPOINT.removesuffix("/responses").rstrip("/") + "/"
    client = OpenAI(
        base_url=base_url,
        api_key=AZURE_VLM_API_KEY,
    )

    p = Path(image_path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")

    r = client.responses.create(
        model=AZURE_VLM_DEPLOYMENT,
        max_output_tokens=300,
        instructions=VLM_SYSTEM_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                {"type": "input_text", "text": question},
            ],
        }],
    )

    return r.output_text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("question")
    args = parser.parse_args()

    print(ask_vlm(args.image, args.question))