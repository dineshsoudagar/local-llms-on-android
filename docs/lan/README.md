# LAN clients and Python scripts

Connect your computer to the model running on your Android phone.

[Back to Pocket LLM](../../README.md) · [LAN demo](../../README.md#lan-server-demo)

- [Start the server](#start-the-server)
- [API endpoints and PowerShell example](#openai-compatible-endpoints)
- [Python client and streaming](#python-client)
- [HTTP script: text, images, and PDFs](#http-script-text-images-and-pdfs)
- [OpenCode](#opencode-coding-agent-backend)
- [Open WebUI and other clients](#open-webui-and-other-clients)
- [Repeated tasks and automation](#a-local-ai-home-server-for-repeated-tasks)

## Start the server

Pocket LLM can expose the selected on-device model to a computer or another device on the same private network. Start the server in the Android app, then open the displayed `/ui` address on your computer to chat in a browser while inference remains on the phone. The same server also exposes an OpenAI-compatible API for scripts and compatible tools.

1. Load a model and open the navigation drawer.
2. Choose **LAN Server**, set a Web UI password of at least eight characters, then tap **Start**. You can use **Generate** for a random password.
3. The **Web UI password** is the simplest credential and works for both browser chat and API clients. A generated API key is optional for advanced clients.
4. From another device on the same network, use the phone's displayed URL with `/v1`, for example `http://PHONE_IP:8080/v1`.

The phone dialog shows the address chosen from the phone's local network interfaces. The computer must be on that same Wi-Fi or LAN; a VPN, guest network, mobile-data address, or a different interface can show a different IP and will not work. This is a private-LAN endpoint, not a public Internet URL.

## OpenAI-compatible endpoints

The API uses `Authorization: Bearer PASSWORD_OR_API_KEY` and exposes:

- `GET /v1/models` - returns the currently loaded model id.
- `GET /v1/models/MODEL_ID` - returns metadata for that model.
- `POST /v1/chat/completions` - accepts standard `messages`, `model`, `stream`, and `max_tokens` fields; `max_tokens` currently reserves context budget rather than hard-capping decoder output.

Example from PowerShell:

```powershell
$base = "http://PHONE_IP:8080/v1"
$password = "YOUR_WEB_UI_PASSWORD"

Invoke-RestMethod "$base/models" -Headers @{ Authorization = "Bearer $password" }

Invoke-RestMethod "$base/chat/completions" -Method Post `
  -Headers @{ Authorization = "Bearer $password" } `
  -ContentType "application/json" `
  -Body (@{
    model = "MODEL_ID_FROM_MODELS"
    messages = @(@{ role = "user"; content = "Explain photosynthesis in one sentence." })
  } | ConvertTo-Json -Depth 5)
```

## Python client

Use [`pocket_llm_openai.py`](../../scripts/pocket_llm_openai.py) with the OpenAI Python library; run these commands from the repository root:

```powershell
python -m pip install openai
$env:POCKET_LLM_BASE_URL = "http://PHONE_IP:8080/v1"
$env:POCKET_LLM_PASSWORD = "YOUR_WEB_UI_PASSWORD"
python scripts/pocket_llm_openai.py "Give me three German words for travel."
python scripts/pocket_llm_openai.py --stream "Write a short greeting."
```

The script discovers the model id automatically. It also accepts `--model`, `--system`, `--base-url`, `--password`, and optional `--api-key` for automation environments.

## HTTP script: text, images, and PDFs

Use [`pocket_llm_http.py`](../../scripts/pocket_llm_http.py) with Python's standard library; run these commands from the repository root:

```powershell
$env:POCKET_LLM_BASE_URL = "http://PHONE_IP:8080/v1"
$env:POCKET_LLM_PASSWORD = "YOUR_WEB_UI_PASSWORD"
python scripts/pocket_llm_http.py "Give me three German words for travel."
python scripts/pocket_llm_http.py --image "C:\path\photo.jpg" "What is in this picture?"
python scripts/pocket_llm_http.py --pdf "C:\path\report.pdf" "Summarize the main findings."
```

- The script discovers the loaded model through `/v1/models` and prints the answer from `/v1/chat/completions`.
- Image and PDF requests upload up to four files through `/ui/attachments` using a temporary chat ID.
- Direct image input requires a compatible model; the browser does not keep a permanent image preview.
- Uploads are processed in the phone's private storage; the script deletes the temporary chat and attachments afterward.
- Original files on your computer remain unchanged; cleanup failures are reported.

## OpenCode coding-agent backend

OpenCode connectivity has been tested against the OpenAI-compatible LiteRT endpoint. Copy [`examples/opencode-pocket-llm.jsonc`](../../examples/opencode-pocket-llm.jsonc) into the OpenCode project configuration, replace `PHONE_IP`, and set the phone's LAN password in `POCKET_LLM_PASSWORD`. The example uses the actual model id `qwen3_litert`; if `GET /v1/models` returns a different selected model id, replace both occurrences in the example.

```powershell
$env:POCKET_LLM_PASSWORD = "YOUR_WEB_UI_PASSWORD"
opencode
```

OpenCode owns tool execution: Pocket LLM returns native LiteRT tool calls with OpenAI-compatible ids and arguments, then accepts the follow-up `role: "tool"` message. ONNX-backed models return an explicit unsupported-backend error for tool requests; `/v1/responses`, Codex integration, and native Ollama routes are not part of this phase.

> OpenCode can connect, but coding-agent workloads often need more context than the validated mobile range. Treat this as tested compatibility, not a recommended long-context coding setup.

## Open WebUI and other clients

For Open WebUI or another OpenAI-compatible client, add a connection with base URL `http://PHONE_IP:8080/v1`, enter the same LAN password as the credential, and select the model id returned by `GET /v1/models`.

Compatible tools can use the password directly:

```bash
curl -X POST http://PHONE_IP:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_WEB_UI_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID_FROM_MODELS","messages":[{"role":"user","content":"Explain photosynthesis in one sentence."}]}'
```

The password is stored on the phone as a salted hash and is accepted as a Bearer credential for the API. The generated API key is optional and can be regenerated when needed. Stop the LAN server before changing the password or model; leaving the password field blank keeps the current password. Android may still stop background work because of device power-management policy, so this is intended for local-network use rather than unattended public hosting.


## A local AI home server for repeated tasks

**The phone can be the model server for your home network, not just a chat screen.** Choose from the built-in LiteRT models or import a compatible `.litertlm` model, then expose the selected model through the same password-protected Web UI and OpenAI-compatible API. You can switch models in the app; the server serves one selected model at a time, and custom-model compatibility remains beta.

Use a script, scheduler, or another connected device to send repeated requests, for example:

- Summarize new local reports or log excerpts on a schedule.
- Classify incoming text or extract fields for a household workflow.
- Ask the model to interpret status updates collected by a monitoring script.

Your automation provides the schedule, data collection, retries, and any alerts; Pocket LLM provides local inference. This can avoid keeping your personal desktop PC running solely for small inference jobs, while keeping the model and its processing on the phone.

For overnight or continuous use, keep the phone powered, check temperature and Android battery restrictions, and test the workload on your device. The LAN service supports screen-off operation while enabled, but **24/7 uptime and power savings have not been benchmarked**; energy use and reliability depend on the phone, model, workload, and network.


[Context and memory guidance](../../README.md#context-length-setting)
