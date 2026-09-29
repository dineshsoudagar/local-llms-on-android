# Pocket LLM v1.6.0

Pocket LLM v1.6.0 lets you use your Android phone as a private local AI server for a computer on the same Wi-Fi or LAN.

## Highlights

- Added a password-protected LAN server with an OpenAI-compatible API.
- Added a browser chat UI at `/ui` so you can chat from a computer while inference stays on the phone.
- Added saved browser conversations, per-chat deletion in the history sidebar, and purple light/dark themes.
- Added Enter to send and Shift+Enter for a new line in the browser composer.
- Added browser PDF and image uploads.
- PDF questions use lightweight on-device BM25 retrieval with embedded-text extraction and OCR fallback for pages that need it. This is intended for focused question answering; retrieval will improve in later releases.
- Browser image input is available when the model selected in the Android app supports direct image input.
- Added a per-model LiteRT context-length input and high-memory warning.
- Added OpenAI-compatible OpenCode connectivity for LiteRT-backed models and native tool-call round trips.
- Improved interrupted model-initialization recovery and context-memory guidance.

## Context guidance

- 8K tokens is the recommended tested baseline.
- 10K to 15K worked in limited manual testing, but has not been broadly validated across devices.
- Qwen LiteRT accepts up to 40K and Gemma LiteRT up to 128K as configurable ceilings. They are not device-safe guarantees.
- A native LiteRT-LM crash was observed with Gemma 4 E2B at 20K on a tested device, consistent with high-context native/GPU memory pressure.

## OpenCode note

OpenCode can connect to Pocket LLM through the OpenAI-compatible LAN endpoint. Coding-agent workloads commonly need more context than the validated mobile range, so this release treats it as tested compatibility rather than a recommended long-context coding setup.

## Files to include in the release

- `pocket_llm_v1.6.0.apk` — Android application package.
- `README.md` — setup, LAN, attachment, and context guidance.
- `examples/opencode-pocket-llm.jsonc` — OpenCode provider configuration example.
