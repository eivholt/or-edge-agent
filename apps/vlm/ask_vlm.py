import base64
import mimetypes
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(override=True)

logfire.configure(service_name="or-edge-agent")
logfire.instrument_openai()

AZURE_VLM_ENDPOINT = os.getenv("AZURE_VLM_ENDPOINT")
AZURE_VLM_DEPLOYMENT = os.getenv("AZURE_VLM_DEPLOYMENT", "gpt-4o")
AZURE_VLM_API_VERSION = os.getenv("AZURE_VLM_API_VERSION", "2024-12-01-preview")
AZURE_VLM_API_KEY = os.getenv("AZURE_VLM_API_KEY")


def image_to_data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@logfire.instrument("ask_vlm question={question}")
def ask_vlm(image_path: str, question: str) -> str:
    client = AzureOpenAI(
        azure_endpoint=AZURE_VLM_ENDPOINT,
        api_key=AZURE_VLM_API_KEY,
        api_version=AZURE_VLM_API_VERSION,
    )

    r = client.chat.completions.create(
        model=AZURE_VLM_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path)
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=300,
    )

    return r.choices[0].message.content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("question")
    args = parser.parse_args()

    print(ask_vlm(args.image, args.question))