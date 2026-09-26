# LittleAgent

A small AI agent for everyday chatting and some light working.

Python runs the agent and exposes it over HTTP; clients talk to it over
server-sent events. A terminal interface and a web frontend are planned.

> Status: the backend works end to end and is covered by tests. The frontend
> directory is scaffolded but empty.

## Layout

```
LittleAgent
|   LICENSE
|   README.md
|
+---backend
|   |   pyproject.toml
|   |
|   +---src
|   |   \---littleagent
|   |       |   __init__.py
|   |       |   main.py          command-line entry point
|   |       |
|   |       +---api              the HTTP layer; imports core
|   |       |       __init__.py
|   |       |       app.py       FastAPI application, CORS policy
|   |       |       routes.py    the SSE endpoint and chunk translation
|   |       |       schemas.py   the wire contract: request and events
|   |       |
|   |       \---core             the agent kernel; imports nothing above it
|   |               __init__.py
|   |               agent.py     model + tools + checkpointer, built once
|   |               tools.py     the tools the agent can call
|   |
|   +---scripts                  run from the backend directory
|   |       __init__.py
|   |       run_all.py           every check in one command
|   |       dump_chunks.py       the chunk shapes a real provider sends
|   |       stream_probe.py      event timing from a running server
|   |
|   \---tests                    pytest suite
|           conftest.py          fakes and chunk builders
|           test_agent.py        agent behaviour, fake model
|           test_api.py          the endpoint and chunk shapes
|           test_tools.py        the tools, as plain functions
|           test_translation.py  chunk -> event translation
|
\---frontend
    \---packages
        +---api-client           shared HTTP client and generated types
        +---tui                  terminal client
        \---web                  Vite + React frontend
```

`frontend/packages/*` are empty placeholders: git does not track directories, so
nothing appears there until the packages have files.

Dependencies point one way. Everything may import `core`, and `core` imports
nothing above it, which is what keeps the agent constructible without a web
server. The HTTP layer reaches the agent factory through the module
(`core_agent.build_agent()`) rather than binding the function at import time, so
that tests replacing it are actually observed.

## Prerequisites

- Python 3.11 or newer. This project is developed against a conda environment at
  `E:\AnacondaEnvs\little-agent` (Python 3.11.16).
- A DeepSeek API key.

## Setup

```powershell
# From the backend directory
pip install -e ".[dev]"
```

Provider integrations are optional, so only the ones you need get installed:

```powershell
pip install -e ".[deepseek]"    # or .[openai], .[anthropic], .[all]
```

### API key

The key is read from the environment, never from a file in the repository:

```powershell
# Windows, persists for the current user
setx DEEPSEEK_API_KEY "sk-..."
```

Environment variables are captured when a process starts. A terminal, an editor
or a server that was already running will not see a key added afterwards - open
a new one.

## Running

```powershell
# From the backend directory
uvicorn littleagent.api.app:app --port 8000 --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API documentation, or
stream a turn from curl:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat/stream `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"What is the weather in San Francisco?\",\"thread_id\":\"t1\"}"
```

The command line entry point runs a single turn without a server:

```powershell
python -m littleagent.main
```

## API

`POST /chat/stream` takes `{"message": str, "thread_id": str}` and answers with
`text/event-stream`. Every frame is `data: <json>` followed by a blank line.

| Event | Meaning |
| --- | --- |
| `token` | a piece of the model's reply |
| `tool_call` | the model asked for a tool |
| `tool_result` | a tool finished |
| `error` | the run failed; the reason is in the payload |
| `done` | the turn is over, always sent last |

`thread_id` selects a conversation. Requests sharing one share their history;
the current checkpointer keeps that history in memory only, so it is lost when
the server restarts.

The definitions live in `backend/src/littleagent/api/schemas.py`, which is the
single source of truth for both the server and every client. `GET /openapi.json`
exposes the same information in a form a TypeScript client can be generated
from.

## Tests

```powershell
# From the backend directory
python -m scripts.run_all              # every check, including real API calls
python -m scripts.run_all --offline-only   # fast, no network, no API key
```

`run_all` reports each layer separately, because each proves something the
others cannot:

1. **offline pytest** - 39 tests, no network. The bulk of the suite.
2. **live pytest** - one real model call, marked `live` and excluded by default.
3. **chunk shapes** - dumps what a real provider actually streams.
4. **streaming over HTTP** - starts a server, streams a turn, checks the timing.

Plain pytest works too:

```powershell
pytest                  # offline suite only
pytest -k tool_call -v  # matching tests, verbose
pytest -m live          # the tests that call the real API
```

### Debugging tools

```powershell
python -m scripts.stream_probe    # event-by-event timing from a running server
python -m scripts.dump_chunks     # the raw chunk shapes a real provider sends
```

`dump_chunks` exists because a fake model lies about shapes. The offline tests
use a fake that delivers each response as one complete message, while a real
provider streams fragments - a bug that dropped every tool call once survived a
fully green test run for exactly that reason. Run `dump_chunks` after changing
the streaming code and compare what it prints against what the tests assume.

## Notes for contributors

- Colour in `run_all` turns itself off when output is redirected, and honours
  `NO_COLOR`. Use `--no-color` to force it off.
- Tests replace the agent via `monkeypatch.setattr(core_agent, "build_agent", ...)`.
  The HTTP layer reaches the factory through the module on purpose, so patching
  `routes.build_agent` would look correct and silently do nothing.
- Comments in this codebase explain why rather than what. The non-obvious
  constraints are called out in place, for example the node filter in
  `frames_from_chunks` and the terminal framing in `translate_events`.
