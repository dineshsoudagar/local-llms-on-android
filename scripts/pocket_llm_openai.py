"""Minimal OpenAI-compatible client for a Pocket LLM phone server.

Install the client once with:
    python -m pip install openai

Then set POCKET_LLM_BASE_URL and POCKET_LLM_PASSWORD, or pass both as flags.
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import APIConnectionError, AuthenticationError, NotFoundError, OpenAI, OpenAIError


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    if value.endswith("/ui"):
        value = value[:-3]
    return value if value.endswith("/v1") else f"{value}/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a text prompt to a Pocket LLM server on the local network."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send; if omitted, read one line from standard input.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("POCKET_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        help="Phone server URL, with or without the /v1 suffix.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("POCKET_LLM_API_KEY"),
        help="Optional generated API key copied from Pocket LLM.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("POCKET_LLM_PASSWORD"),
        help="LAN password set in Pocket LLM; this is the simplest credential.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("POCKET_LLM_MODEL"),
        help="Model id; if omitted, use the first id returned by /v1/models.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system instruction for this request.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print the answer as the phone generates it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    credential = args.password or args.api_key or input("Pocket LLM password: ").strip()
    prompt = args.prompt or input("Prompt: ").strip()

    if not credential:
        print("A LAN password or API key is required.", file=sys.stderr)
        return 2
    if not prompt:
        print("A prompt is required.", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=credential,
        base_url=normalize_base_url(args.base_url),
    )
    model = args.model
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    try:
        if not model:
            models = client.models.list()
            if not models.data:
                print("Pocket LLM did not return a model id.", file=sys.stderr)
                return 1
            model = models.data[0].id

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=args.stream,
        )
    except AuthenticationError:
        print(
            "Authentication failed. Use the LAN password set in the phone app, "
            "or the optional generated API key.",
            file=sys.stderr,
        )
        return 1
    except NotFoundError:
        print(
            f"API endpoint not found at {normalize_base_url(args.base_url)}. "
            "Use the phone address without /ui, ending in /v1.",
            file=sys.stderr,
        )
        return 1
    except APIConnectionError as error:
        print(
            "Could not reach the phone. Check that the phone server is running "
            "and both devices are on the same network.",
            file=sys.stderr,
        )
        print(f"Details: {error}", file=sys.stderr)
        return 1
    except OpenAIError as error:
        print(f"Pocket LLM request failed: {error}", file=sys.stderr)
        return 1

    if args.stream:
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        print(response.choices[0].message.content or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
