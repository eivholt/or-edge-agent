import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8001/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-key")


def image_to_data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def ask_vlm(image_path: str, question: str) -> str:
    client = OpenAI(base_url=VLM_BASE_URL, api_key=OPENAI_API_KEY)

    r = client.chat.completions.create(
        model=VLM_MODEL,
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
        max_tokens=300
    )

    return r.choices[0].message.content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("question")
    args = parser.parse_args()

    print(ask_vlm(args.image, args.question))