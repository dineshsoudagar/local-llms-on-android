"""Pocket LLM LAN API example for text, images, and PDFs using Python's standard library."""

from __future__ import annotations

import argparse
import getpass
import json
import mimetypes
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4


def server_root(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/ui"):
        base = base[:-3]
    if not base.endswith("/v1"):
        base += "/v1"
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Set --base-url to the phone's LAN address, such as http://PHONE_IP:8080/v1.")
    return base[:-3]


def api_url(base_url: str, path: str) -> str:
    return server_root(base_url) + "/v1" + path


def request_json(
    url: str,
    credential: str,
    payload: dict | bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if isinstance(payload, dict) else payload
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {credential}",
            **({"Content-Type": "application/json"} if isinstance(payload, dict) else {}),
            **(extra_headers or {}),
        },
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            detail = json.load(error).get("error", {}).get("message", error.reason)
        except (ValueError, AttributeError):
            detail = error.reason
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the phone: {error.reason}") from error


def upload_file(base_url: str, credential: str, chat_id: str, path: Path, kind: str) -> str:
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    size_limit = 12 * 1024 * 1024 if kind == "image" else 16 * 1024 * 1024
    if path.stat().st_size > size_limit:
        raise ValueError(f"{path.name} exceeds the {size_limit // (1024 * 1024)} MiB upload limit.")
    content_type = "application/pdf" if kind == "pdf" else mimetypes.guess_type(path.name)[0]
    if kind == "image" and (not content_type or not content_type.startswith("image/")):
        raise ValueError(f"Expected an image file: {path}")
    result = request_json(
        server_root(base_url) + "/ui/attachments",
        credential,
        path.read_bytes(),
        {
            "Content-Type": content_type,
            "X-Chat-Id": chat_id,
            "X-Upload-Name": quote(path.name),
        },
    )
    return result["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Text prompt; reads stdin or asks if omitted.")
    parser.add_argument("--base-url", default=os.getenv("POCKET_LLM_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("POCKET_LLM_MODEL"), help="Defaults to the loaded model.")
    parser.add_argument("--system", help="Optional system instruction.")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Image to ask about; repeat for more images.")
    parser.add_argument("--pdf", action="append", type=Path, default=[], help="PDF to ask about; repeat for more PDFs.")
    args = parser.parse_args()

    if not args.base_url:
        parser.error("Set POCKET_LLM_BASE_URL or pass --base-url with the phone's LAN address.")
    credential = os.getenv("POCKET_LLM_PASSWORD") or os.getenv("POCKET_LLM_API_KEY")
    if not credential:
        credential = getpass.getpass("Pocket LLM LAN password: ").strip()
    if not credential:
        parser.error("A LAN password or API key is required.")

    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read().strip() if not sys.stdin.isatty() else input("Prompt: ").strip()
    if not prompt:
        parser.error("A text prompt is required.")
    files = [(path, "image") for path in args.image] + [(path, "pdf") for path in args.pdf]
    if len(files) > 4:
        parser.error("Attach at most four files per request.")

    chat_id = "cli_" + uuid4().hex if files else None
    failed = False
    try:
        models_url = api_url(args.base_url, "/models")
        chat_url = api_url(args.base_url, "/chat/completions")
        model = args.model
        if not model:
            models = request_json(models_url, credential).get("data", [])
            if not models:
                raise RuntimeError("The phone did not report a loaded model.")
            model = models[0]["id"]
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages}
        if chat_id:
            payload["chat_id"] = chat_id
            payload["attachments"] = [
                upload_file(args.base_url, credential, chat_id, path, kind)
                for path, kind in files
            ]
        response = request_json(chat_url, credential, payload)
        print(response["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        print(f"Pocket LLM request failed: {error}", file=sys.stderr)
        failed = True
    finally:
        if chat_id:
            try:
                request_json(server_root(args.base_url) + "/ui/chats/delete", credential, {"id": chat_id})
            except (ValueError, RuntimeError) as error:
                print(f"Could not remove temporary phone attachments: {error}", file=sys.stderr)
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
