import os
import subprocess
import time
from pathlib import Path

import ollama
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from three_layer_firewall.db.logger import RequestLogger
from three_layer_firewall.db.session import make_engine, make_session_factory
from three_layer_firewall.firewall import Firewall
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_PATH = ROOT / "models" / "classifier-model"
app = FastAPI(title="LLM Firewall API")

class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful AI assistant."
    protected_terms: list[str] = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing firewall...")
firewall = Firewall(
    system_prompt="You are a helpful AI assistant.",
    model_path=str(MODEL_PATH)
)
print("Firewall ready.")

def get_firewall_version() -> str:
    """Git commit the server is running, so every logged row can be tied to the code."""
    try:
        run = lambda *cmd: subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        version = run("git", "rev-parse", "--short", "HEAD")
        return version + ("-dirty" if run("git", "status", "--porcelain") else "")
    except Exception:
        return "unknown"

LOG_TO_DB = os.getenv("LOG_TO_DB", "true").lower() == "true"
STORE_TEXT = os.getenv("LOG_PROMPT_TEXT", "false").lower() == "true"
RUN_LABEL = os.getenv("RUN_LABEL")      # optional tag for an experiment, e.g. "demo-1"

request_logger = RequestLogger(
    make_session_factory(make_engine()) if LOG_TO_DB else None,
    store_text=STORE_TEXT,
    firewall_version=get_firewall_version(),
    config={
        "clf_threshold": Firewall.ATTACK_THRESHOLD,
        "max_length": Firewall.MAX_LENGTH,
        "classifier": MODEL_PATH.name,
    },
)

class ScanRequest(BaseModel):
    prompt: str

class ScanResponse(BaseModel):
    blocked: bool
    layer: str
    reason: str
    confidence: float | None

@app.get("/")
def root():
    return {"status": "LLM Firewall API is running"}

@app.post("/scan")
def scan(request: ScanRequest):
    t0 = time.perf_counter()
    result = firewall.scan_input(request.prompt)
    total_ms = (time.perf_counter() - t0) * 1000

    request_logger.log(
        endpoint="/scan",
        prompt=request.prompt,
        scan_result=result,
        total_ms=total_ms,
        run_label=RUN_LABEL,
    )
    return {
        "blocked": result["blocked"],
        "layer": result["layer"],
        "reason": result["reason"],
        "confidence": result.get("confidence")
    }

LLM_MODEL = "llama3.2"

@app.post("/chat")
def chat(request: ChatRequest):
    t0 = time.perf_counter()
    scan_result = firewall.scan_input(request.prompt)

    def save(**extra):
        """Write one log row for this request; extra = LLM / output-filter fields."""
        request_logger.log(
            endpoint="/chat",
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            scan_result=scan_result,
            total_ms=(time.perf_counter() - t0) * 1000,
            run_label=RUN_LABEL,
            **extra,
        )

    # blocked by layer 1 or 2: the LLM is never called
    if scan_result["blocked"]:
        save()
        return {
            "blocked": True,
            "layer": scan_result["layer"],
            "reason": scan_result["reason"],
            "confidence": scan_result.get("confidence"),
            "response": None
        }

    # passed — send to Llama3
    t1 = time.perf_counter()
    try:
        result = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt}
            ]
        )
    except Exception as e:
        # the log marks it so it is not counted as a real "passed"
        save(llm_model=LLM_MODEL, llm_ms=(time.perf_counter() - t1) * 1000,
             status="llm_error", error=str(e))

        return JSONResponse(
            status_code=502,
            content={
                "blocked": False,
                "layer": "llm_error",
                "reason": "LLM backend unavailable",
                "confidence": None,
                "response": None,
            },
        )
    llm_ms = (time.perf_counter() - t1) * 1000
    response_text = result["message"]["content"]

    # layer 3
    output_result = firewall.scan_output(response_text, request.system_prompt, request.protected_terms)
    save(llm_model=LLM_MODEL, llm_response=response_text, llm_ms=llm_ms,
         output_result=output_result)

    if output_result["blocked"]:
        return {
            "blocked": True,
            "layer": "output_filter",
            "reason": output_result["reason"],
            "confidence": output_result.get("confidence"),
            "response": None
        }

    return {
        "blocked": False,
        "layer": "all_passed",
        "reason": "passed all firewall layers",
        "confidence": scan_result.get("confidence"),
        "response": response_text
    }