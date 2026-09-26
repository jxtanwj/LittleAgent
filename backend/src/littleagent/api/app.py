"""FastAPI application: wires the routes and the CORS policy, then uvicorn serves it.

Run it with:
    uvicorn littleagent.api.app:app --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from littleagent.api.routes import router

app = FastAPI(
    title="LittleAgent",
    description="Streams agent turns to clients over server-sent events.",
    version="0.1.0",
)

# CORS only matters for browsers. Native clients - the Python CLI, a Node TUI -
# are not subject to it, so a 403 here never shows up in those.
#
# During development Vite serves the frontend from :5173 while this app runs on
# :8000, which makes every request cross-origin. Two ways out: proxy /api through
# Vite so the browser sees one origin and CORS never applies, or allow the origin
# here. Both are listed because the proxy is easy to forget when someone opens
# the API from a plain page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",   # Vite preview build
        "http://127.0.0.1:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Interactive docs come for free at /docs, and the raw schema that a TypeScript
# client generator would consume is at /openapi.json.