import logging
import uuid
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from agenticai import app as graph_app  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("math_solver.api")

# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(title="Math Solver Agent", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# In-memory session store  {session_id: {status, state, thread_id, ...}}
sessions: dict = {}


# ── Request logging middleware ─────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.now()
    response = await call_next(request)
    elapsed = (datetime.now() - start).total_seconds()
    logger.info("%s %s → %s (%.2fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/solve")
async def solve(
    text_input: str = Form(""),
    image_file: UploadFile = File(None),
):
    """Kick off the multi-agent graph. Returns a session_id to poll."""
    image_bytes = None
    if image_file is not None and image_file.filename:
        image_bytes = await image_file.read()
        logger.info("Received image upload: %s (%d bytes)", image_file.filename, len(image_bytes))

    session_id = str(uuid.uuid4())[:8]
    thread_id = f"session-{session_id}"

    logger.info("New solve session %s — text=%s, has_image=%s",
                session_id, bool(text_input), bool(image_bytes))

    sessions[session_id] = {
        "status": "running",
        "current_step": "parser",
        "thread_id": thread_id,
        "state": None,
        "need_clarification": False,
        "error": None,
    }

    # Run graph in background thread (it's sync/blocking)
    asyncio.get_event_loop().run_in_executor(
        None,
        _run_graph,
        session_id,
        thread_id,
        text_input,
        image_bytes,
    )

    return JSONResponse({"session_id": session_id, "status": "running"})


def _run_graph(session_id: str, thread_id: str, text_input: str, image_bytes: Optional[bytes]):
    """Run the LangGraph in a background thread."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "agent_input": {
                "text_input": text_input or None,
                "image_input": bytearray(image_bytes) if image_bytes else None,
            }
        }

        result = graph_app.invoke(state, config=config)

        # Check if graph was interrupted (HITL)
        snapshot = graph_app.get_state(config)
        if snapshot.next:
            logger.info("Session %s interrupted for HITL at %s", session_id, snapshot.next)
            sessions[session_id].update({
                "status": "need_clarification",
                "need_clarification": True,
                "current_step": "awaiting_clarification",
                "state": result,
            })
        else:
            logger.info("Session %s completed successfully", session_id)
            sessions[session_id].update({
                "status": "completed",
                "current_step": "done",
                "state": result,
            })
    except Exception as e:
        logger.exception("Session %s failed", session_id)
        sessions[session_id].update({
            "status": "error",
            "error": str(e),
        })


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Poll for the current state of a solve session."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    state = session.get("state") or {}
    
    # Clean state for JSON serialization (remove bytes)
    clean_state = {}
    for key, val in state.items():
        if key == "agent_input":
            clean_state[key] = {
                "text_input": val.get("text_input") if isinstance(val, dict) else None,
                "has_image": bool(val.get("image_input")) if isinstance(val, dict) else False,
            }
        elif isinstance(val, (dict, list, str, int, float, bool, type(None))):
            clean_state[key] = val
        else:
            clean_state[key] = str(val)

    return JSONResponse({
        "session_id": session_id,
        "status": session["status"],
        "current_step": session.get("current_step"),
        "need_clarification": session.get("need_clarification", False),
        "state": clean_state,
        "logs": clean_state.get("logs", []),
        "error": session.get("error"),
    })


@app.post("/proceed/{session_id}")
async def proceed(session_id: str):
    """User approves the parsed problem — resume the graph."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if session["status"] != "need_clarification":
        return JSONResponse({"error": "Session not in clarification state"}, status_code=400)

    logger.info("Session %s — user chose PROCEED", session_id)
    session["status"] = "running"
    session["need_clarification"] = False
    session["current_step"] = "solver"

    asyncio.get_event_loop().run_in_executor(
        None, _resume_graph, session_id, session["thread_id"], "proceed"
    )

    return JSONResponse({"status": "running", "action": "proceed"})


@app.post("/retry/{session_id}")
async def retry(session_id: str):
    """User wants to re-parse — resume the graph with retry."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if session["status"] != "need_clarification":
        return JSONResponse({"error": "Session not in clarification state"}, status_code=400)

    logger.info("Session %s — user chose RETRY", session_id)
    session["status"] = "running"
    session["need_clarification"] = False
    session["current_step"] = "parser"

    asyncio.get_event_loop().run_in_executor(
        None, _resume_graph, session_id, session["thread_id"], "retry"
    )

    return JSONResponse({"status": "running", "action": "retry"})


def _resume_graph(session_id: str, thread_id: str, action: str):
    """Resume graph from HITL interrupt."""
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Update state with user's decision
        graph_app.update_state(config, {"clarification_action": action})

        # Resume
        result = graph_app.invoke(None, config=config)

        # Check for another interrupt
        snapshot = graph_app.get_state(config)
        if snapshot.next:
            sessions[session_id].update({
                "status": "need_clarification",
                "need_clarification": True,
                "current_step": "awaiting_clarification",
                "state": result,
            })
        else:
            sessions[session_id].update({
                "status": "completed",
                "current_step": "done",
                "state": result,
            })
    except Exception as e:
        logger.exception("Session %s resume failed", session_id)
        sessions[session_id].update({
            "status": "error",
            "error": str(e),
        })