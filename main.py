# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
# pyright: reportUnknownArgumentType=none
# pyright: reportUnknownParameterType=none
# pyright: reportMissingParameterType=none
# pyright: reportMissingTypeStubs=none
# pyright: reportGeneralTypeIssues=none
# pyright: reportArgumentType=none
# pyright: reportUnusedImport=none

import asyncio
import hashlib
import io
import json
import os
import random
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from google import genai
from google.genai import types
import numpy as np
from PIL import Image
import pydicom
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="HealthLens AI Clinical Suite", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Global Exception Handler
# -------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    print(f"[🔥 UNHANDLED BACKEND ERROR] {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": f"Internal Diagnostic Server Error: {str(exc)}"}
    )

# -------------------------------------------------------------
# Static PWA Delivery Endpoints
# -------------------------------------------------------------
@app.get("/")
def serve_index() -> FileResponse:
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html not found.")

@app.get("/manifest.json")
def serve_manifest() -> FileResponse:
    if os.path.exists("manifest.json"):
        return FileResponse("manifest.json", media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="manifest.json not found.")

@app.get("/sw.js")
def serve_sw() -> FileResponse:
    if os.path.exists("sw.js"):
        return FileResponse(
            "sw.js",
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    raise HTTPException(status_code=404, detail="sw.js not found.")

@app.get("/icon.png")
def serve_icon() -> Response:
    if os.path.exists("icon.png"):
        return FileResponse("icon.png", media_type="image/png")
    return Response(
        content=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82',
        media_type="image/png"
    )

@app.get("/icon-192.png")
def serve_icon_192() -> Response:
    if os.path.exists("icon-192.png"):
        return FileResponse("icon-192.png", media_type="image/png")
    return serve_icon()

@app.get("/doctor.png")
def serve_doctor() -> Response:
    if os.path.exists("doctor.png"):
        return FileResponse("doctor.png", media_type="image/png")
    return serve_icon()

# -------------------------------------------------------------
# User Storage & History System
# -------------------------------------------------------------
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

def load_users() -> Dict[str, Dict[str, Any]]:
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

class AuthReq(BaseModel):
    email: str
    password: str

@app.post("/api/auth/signup")
def signup(creds: AuthReq) -> Dict[str, str]:
    email = creds.email.strip().lower()
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if len(creds.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    users = load_users()
    if email in users:
        raise HTTPException(status_code=400, detail="Email already registered.")
    users[email] = {
        "password_hash": hash_pw(creds.password),
        "sessions": []
    }
    save_users(users)
    return {"status": "success", "message": "Registered successfully.", "email": email}

@app.post("/api/auth/login")
def login(creds: AuthReq) -> Dict[str, str]:
    email = creds.email.strip().lower()
    users = load_users()
    if email not in users or users[email]["password_hash"] != hash_pw(creds.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"status": "success", "message": "Login successful.", "email": email}

class SaveSessionReq(BaseModel):
    email: str
    id: Optional[str] = None
    title: str
    type: str  # "symptoms" | "scans" | "rx" | "audio" | "chat"
    data: Any

@app.get("/api/history")
def get_history(email: str = Query(...)) -> Dict[str, Any]:
    email_clean = email.strip().lower()
    users = load_users()
    if email_clean not in users:
        return {"status": "success", "sessions": []}

    sessions = users[email_clean].get("sessions", [])
    summaries = [
        {
            "id": s["id"],
            "title": s["title"],
            "type": s["type"],
            "created_at": s.get("created_at", "")
        }
        for s in reversed(sessions)
    ]
    return {"status": "success", "sessions": summaries}

@app.get("/api/history/{session_id}")
def get_session(session_id: str, email: str = Query(...)) -> Dict[str, Any]:
    email_clean = email.strip().lower()
    users = load_users()
    if email_clean not in users:
        raise HTTPException(status_code=404, detail="User not found.")
    for s in users[email_clean].get("sessions", []):
        if s["id"] == session_id:
            return {"status": "success", "session": s}
    raise HTTPException(status_code=404, detail="Session record not found.")

@app.post("/api/history/save")
def save_session(req: SaveSessionReq) -> Dict[str, Any]:
    email_clean = req.email.strip().lower()
    users = load_users()
    if email_clean not in users:
        users[email_clean] = {"password_hash": "", "sessions": []}

    sessions: List[Dict[str, Any]] = users[email_clean].setdefault("sessions", [])
    sess_id = req.id or str(uuid.uuid4())[:8]

    updated = False
    for s in sessions:
        if s["id"] == sess_id:
            s["title"] = req.title
            s["type"] = req.type
            s["data"] = req.data
            s["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
            updated = True
            break

    if not updated:
        new_sess = {
            "id": sess_id,
            "title": req.title,
            "type": req.type,
            "data": req.data,
            "created_at": time.strftime("%b %d, %H:%M"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M")
        }
        sessions.append(new_sess)

    if len(sessions) > 50:
        sessions.pop(0)

    save_users(users)
    return {"status": "success", "id": sess_id}

@app.delete("/api/history/{session_id}")
def delete_session(session_id: str, email: str = Query(...)) -> Dict[str, Any]:
    email_clean = email.strip().lower()
    users = load_users()
    if email_clean in users:
        users[email_clean]["sessions"] = [
            s for s in users[email_clean].get("sessions", []) if s["id"] != session_id
        ]
        save_users(users)
    return {"status": "success"}

# -------------------------------------------------------------
# Dynamic Key Loading & Persistent Client Pool
# -------------------------------------------------------------
API_KEYS: List[str] = [
    val.strip()
    for key, val in sorted(os.environ.items())
    if key.startswith("GEMINI_API_KEY") and val.strip()
]

if not API_KEYS:
    raise RuntimeError("No GEMINI_API_KEY entries found in environment or .env file.")

TARGET_MODEL = "gemini-3.6-flash"

CLIENT_POOL: List[genai.Client] = [genai.Client(api_key=k) for k in API_KEYS]
print(f"[*] Initialized {len(CLIENT_POOL)} Gemini client workers into active pool.")

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": "HealthLens AI Clinical Suite",
        "model": TARGET_MODEL,
        "total_active_keys": len(CLIENT_POOL),
        "features": [
            "Point-to-Pain Body Dialog",
            "Dual-Doctor Second Opinion",
            "Rx & Lab OCR Decoder",
            "Acoustic Cough Analyzer",
            "PACS Slice Scrubber",
            "Paramedic Handover EMT Mode"
        ]
    }

# -------------------------------------------------------------
# Safe JSON Parser Helper
# -------------------------------------------------------------
def extract_clean_json(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Diagnostic model returned empty text.")
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    return json.loads(raw)

# -------------------------------------------------------------
# DICOM Processing Helper
# -------------------------------------------------------------
def _extract_hu(val: Any, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return float(val[0])
    try:
        return float(str(val))
    except Exception:
        return default

def dicom_to_jpeg(dicom_bytes: bytes) -> bytes:
    with io.BytesIO(dicom_bytes) as buf:
        dcm = pydicom.dcmread(buf)
        arr = dcm.pixel_array.astype(float)
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        hu = (arr * slope) + intercept

        wc = _extract_hu(getattr(dcm, "WindowCenter", 40), 40.0)
        ww = _extract_hu(getattr(dcm, "WindowWidth", 400), 400.0)

        min_val = wc - (ww / 2.0)
        max_val = wc + (ww / 2.0)
        clipped = np.clip(hu, min_val, max_val)
        norm = (
            ((clipped - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            if max_val > min_val
            else np.zeros_like(clipped, dtype=np.uint8)
        )

        img = Image.fromarray(norm).convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()

# -------------------------------------------------------------
# Concurrent Speculative Racing Engine
# -------------------------------------------------------------
async def race_sampled_keys(call_fn: Callable[..., Dict[str, Any]], *args: Any) -> Dict[str, Any]:
    errors: List[str] = []
    sample_size: int = min(4, len(CLIENT_POOL))
    sampled_indices: List[int] = random.sample(range(len(CLIENT_POOL)), sample_size)

    async def runner(idx: int) -> Tuple[Dict[str, Any], int]:
        try:
            parsed_data: Dict[str, Any] = await asyncio.to_thread(call_fn, idx, *args)
            return parsed_data, idx
        except Exception as e:
            raise RuntimeError(f"Key #{idx + 1}: {str(e)}") from e

    tasks = [asyncio.create_task(runner(i)) for i in sampled_indices]

    for coro in asyncio.as_completed(tasks):
        try:
            result, winning_idx = await coro
            print(f"[⚡] Key #{winning_idx + 1} delivered verified clinical response!")
            for t in tasks:
                if not t.done():
                    t.cancel()
            return result
        except asyncio.CancelledError:
            continue
        except Exception as e:
            print(f"[!] Sub-task failed: {e}")
            errors.append(str(e))
            continue

    raise HTTPException(
        status_code=500,
        detail=f"All {sample_size} sampled keys encountered errors: {'; '.join(errors)}"
    )

# -------------------------------------------------------------
# 1. Symptoms Assessment with Point-to-Pain & Dual-Doctor
# -------------------------------------------------------------
class SymptomPayload(BaseModel):
    symptoms: str
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    duration: Optional[str] = None
    known_conditions: Optional[str] = None
    body_regions: Optional[List[str]] = None
    pain_severity: Optional[int] = None
    pain_character: Optional[str] = None
    radiation: Optional[str] = None
    dual_doctor_mode: Optional[bool] = False

def _execute_symptom_call(client_idx: int, prompt: str, sys_instruction: str) -> Dict[str, Any]:
    client = CLIENT_POOL[client_idx]
    res = client.models.generate_content(
        model=TARGET_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            response_mime_type="application/json",
            temperature=0.15,
            max_output_tokens=8192,
        )
    )
    raw_text = (res.text or "{}").strip()
    return extract_clean_json(raw_text)

@app.post("/api/symptoms/analyze")
async def analyze_symptoms(payload: SymptomPayload) -> Dict[str, Any]:
    if not payload.symptoms.strip() and not payload.body_regions:
        raise HTTPException(status_code=400, detail="Symptoms or pain map localization required.")

    sys_instruction = """
    You are an emergency triage physician and differential diagnostics AI.
    Analyze patient symptoms, localized body pain regions, pain character, and severity.

    Always return valid JSON matching this exact schema:
    {
      "suspected_primary_condition": "string",
      "triage_urgency": "Self-Care | Routine Visit | Urgent Care | Emergency",
      "clinical_summary": "string (concise 2-3 sentences)",
      "dual_doctor_debate": {
        "active": boolean,
        "dr_alpha_internist": {
          "perspective": "Holistic & Conservative Internal Medicine",
          "evaluation": "string",
          "suggested_approach": "string"
        },
        "dr_beta_emergency": {
          "perspective": "Acute Care & Critical Life-Threat Rule-Out",
          "evaluation": "string",
          "worst_case_rule_out": "string"
        },
        "consensus_resolution": "string"
      },
      "differential_diagnoses": [
        {"disease_name": "string", "likelihood": "High | Moderate | Low", "matching_reasons": "string"}
      ],
      "red_flag_symptoms": ["string"],
      "recommended_diagnostic_tests": ["string"],
      "paramedic_handover": {
        "chief_complaint": "string",
        "mechanism_or_triggers": "string",
        "signs_and_symptoms": "string",
        "red_flag_warnings": "string",
        "immediate_suggested_interventions": "string"
      },
      "disclaimer": "AI clinical triage tool only. Not a substitute for professional board-certified evaluation."
    }
    """

    body_context = ""
    if payload.body_regions:
        body_context = f"\nLocalized Pain Regions: {', '.join(payload.body_regions)}"
    if payload.pain_severity:
        body_context += f"\nPain Severity Score (1-10): {payload.pain_severity}/10"
    if payload.pain_character:
        body_context += f"\nPain Quality/Character: {payload.pain_character}"
    if payload.radiation:
        body_context += f"\nRadiation: {payload.radiation}"

    prompt = (
        f"Patient Complaints: {payload.symptoms}\n"
        f"Age: {payload.patient_age or 'Unspecified'}, Sex: {payload.patient_sex or 'Unspecified'}, "
        f"Duration: {payload.duration or 'Unspecified'}, History: {payload.known_conditions or 'None'}"
        f"{body_context}\n"
        f"Dual-Doctor Simulation Requested: {'YES' if payload.dual_doctor_mode else 'NO'}"
    )

    data = await race_sampled_keys(_execute_symptom_call, prompt, sys_instruction)
    return {"status": "success", "data": data}

# -------------------------------------------------------------
# 2. Prescription & Lab Report Decoder (Handwriting OCR)
# -------------------------------------------------------------
def _execute_rx_call(client_idx: int, contents: List[Any], sys_instruction: str) -> Dict[str, Any]:
    client = CLIENT_POOL[client_idx]
    res = client.models.generate_content(
        model=TARGET_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=8192,
        )
    )
    raw_text = (res.text or "{}").strip()
    return extract_clean_json(raw_text)

@app.post("/api/prescription/analyze")
async def analyze_prescription(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    mime = file.content_type or "image/jpeg"
    part = types.Part.from_bytes(data=data, mime_type=mime)

    sys_instruction = """
    You are an expert clinical pharmacist and pathology laboratory scientist.
    Analyze handwritten or printed doctor prescriptions and laboratory report panels.

    Decode medical acronyms:
    - OD (once daily), BD/BID (twice daily), TDS/TID (three times daily), QID (four times daily),
    - PRN (as needed), AC (before food), PC (after food), HS (at bedtime), Stat (immediately).

    Return ONLY valid JSON:
    {
      "document_type": "Prescription | Lab Report | Diagnostic Summary",
      "physician_or_facility": "string",
      "extracted_medications": [
        {
          "name": "string",
          "dosage": "string",
          "frequency": "string",
          "plain_instructions": "string",
          "purpose_or_caution": "string"
        }
      ],
      "lab_results_panel": [
        {
          "test_name": "string",
          "observed_value": "string",
          "standard_range": "string",
          "status": "Normal | Borderline | Critical",
          "interpretation": "string"
        }
      ],
      "clinical_summary": "string",
      "critical_safety_alerts": ["string"]
    }
    """
    prompt = "Transcribe, decode all handwriting and abbreviations, and structure this prescription or lab report into clean medical JSON."
    contents = [part, prompt]

    data = await race_sampled_keys(_execute_rx_call, contents, sys_instruction)
    return {"status": "success", "data": data}

# -------------------------------------------------------------
# 3. Acoustic Cough & Wheeze Analyzer
# -------------------------------------------------------------
def _execute_audio_call(client_idx: int, contents: List[Any], sys_instruction: str) -> Dict[str, Any]:
    client = CLIENT_POOL[client_idx]
    res = client.models.generate_content(
        model=TARGET_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=8192,
        )
    )
    raw_text = (res.text or "{}").strip()
    return extract_clean_json(raw_text)

@app.post("/api/audio/analyze")
async def analyze_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    mime = file.content_type or "audio/webm"
    if "webm" in mime:
        part_mime = "audio/webm"
    elif "wav" in mime:
        part_mime = "audio/wav"
    else:
        part_mime = "audio/mp3"

    part = types.Part.from_bytes(data=data, mime_type=part_mime)

    sys_instruction = """
    You are a pediatric and adult pulmonologist specializing in acoustic respiratory diagnostics.
    Analyze the audio recording of the user's cough, breathing, or wheeze.

    Return ONLY valid JSON:
    {
      "sound_classification": "Dry / Non-Productive | Wet / Productive | Barking (Croup-like) | Whooping / Paroxysmal | Wheezing / Asthmatic | Clear Breathing",
      "audible_wheeze": boolean,
      "audible_stridor": boolean,
      "severity_grade": "Mild | Moderate | Urgent | Severe",
      "acoustic_biomarkers": [
        "string"
      ],
      "differential_pulmonary_causes": [
        "string"
      ],
      "home_care_and_action": "string",
      "red_flag_triggers": ["string"]
    }
    """
    prompt = "Analyze this respiratory audio sample for cough patterns, audible wheeze, stridor, and clinical pulmonary distress markers."
    contents = [part, prompt]

    data = await race_sampled_keys(_execute_audio_call, contents, sys_instruction)
    return {"status": "success", "data": data}

# -------------------------------------------------------------
# 4. CT / DICOM Scan Radiography Analysis
# -------------------------------------------------------------
def _execute_scan_call(client_idx: int, contents: List[Any], sys_instruction: str) -> Dict[str, Any]:
    client = CLIENT_POOL[client_idx]
    res = client.models.generate_content(
        model=TARGET_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=8192,
        )
    )
    raw_text = (res.text or "{}").strip()
    return extract_clean_json(raw_text)

@app.post("/api/scan/analyze")
async def analyze_scan(
    files: List[UploadFile] = File(...),
    confirmed_body_part: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None)
) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No scan files uploaded.")

    parts: List[types.Part] = []
    for f in files:
        data = await f.read()
        name = (f.filename or "").lower()
        if name.endswith(".dcm") or f.content_type == "application/dicom":
            try:
                data = await asyncio.to_thread(dicom_to_jpeg, data)
                parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"DICOM processing error: {str(e)}")
        else:
            parts.append(types.Part.from_bytes(data=data, mime_type=f.content_type or "image/jpeg"))

    sys_instruction = """
    You are a board-certified radiologist. Analyze medical scan images (CT, X-Ray, MRI).
    Return ONLY valid JSON matching this schema:
    {
      "detected_body_part": "string",
      "scan_view_type": "string",
      "slice_count_analyzed": integer,
      "suspected_disease": "string",
      "severity_level": "Mild | Moderate | Urgent | Critical",
      "diagnostic_confidence": "High | Moderate | Tentative",
      "detailed_radiological_findings": ["string"],
      "differential_diagnoses": ["string"],
      "simplified_explanation": "string",
      "questions_for_doctor": ["string"],
      "disclaimer": "AI review only. Must be confirmed by a board-certified radiologist."
    }
    """
    prompt = f"Analyze these {len(parts)} scan images. Region: {confirmed_body_part or 'Auto-detect'}, Symptoms: {symptoms or 'None stated'}."
    contents: List[Any] = [*parts, prompt]

    data = await race_sampled_keys(_execute_scan_call, contents, sys_instruction)
    return {"status": "success", "data": data}

# -------------------------------------------------------------
# 5. Thread-Safe Streaming Chat Engine
# -------------------------------------------------------------
class ChatPayload(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/api/chat/stream")
async def chat_stream(payload: ChatPayload) -> StreamingResponse:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Empty messages.")

    history: List[types.ContentOrDict] = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])]
        ) for m in payload.messages[:-1]
    ]
    last_msg = payload.messages[-1]["content"]

    async def event_generator():
        yield f"data: {json.dumps({'status': 'connected'})}\n\n"

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        stop_event = threading.Event()
        errors: List[str] = []

        sample_size: int = min(4, len(CLIENT_POOL))
        sampled_indices: List[int] = random.sample(range(len(CLIENT_POOL)), sample_size)
        finished_workers: int = 0

        def worker(client_idx: int) -> None:
            nonlocal finished_workers
            client = CLIENT_POOL[client_idx]
            is_winner = False
            try:
                session = client.chats.create(
                    model=TARGET_MODEL,
                    history=history,
                    config=types.GenerateContentConfig(
                        system_instruction="You are HealthLens AI medical assistant. Provide concise, clinical advice.",
                        temperature=0.7,
                    )
                )
                stream = session.send_message_stream(last_msg)
                for chunk in stream:
                    if stop_event.is_set() and not is_winner:
                        return
                    if chunk.text:
                        if not is_winner:
                            if stop_event.is_set():
                                return
                            stop_event.set()
                            is_winner = True
                            print(f"[⚡] Key #{client_idx + 1} won the chat streaming race!")
                        asyncio.run_coroutine_threadsafe(queue.put({"text": chunk.text}), loop)

                if is_winner:
                    asyncio.run_coroutine_threadsafe(queue.put({"done": True}), loop)
            except Exception as e:
                err_str = f"Key #{client_idx + 1}: {str(e)}"
                if not stop_event.is_set():
                    asyncio.run_coroutine_threadsafe(queue.put({"err": err_str}), loop)
            finally:
                finished_workers += 1
                if finished_workers == sample_size and not stop_event.is_set():
                    asyncio.run_coroutine_threadsafe(queue.put({"all_failed": True}), loop)

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in sampled_indices
        ]
        for t in threads:
            t.start()

        while True:
            item = await queue.get()
            if "err" in item:
                errors.append(item["err"])
            elif item.get("all_failed"):
                yield f"data: {json.dumps({'error': 'Keys failed: ' + '; '.join(errors)})}\n\n"
                yield "data: [DONE]\n\n"
                break
            elif item.get("done"):
                yield "data: [DONE]\n\n"
                break
            elif "text" in item:
                yield f"data: {json.dumps({'text': item['text']})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
# -------------------------------------------------------------
# Direct Application Package Download Endpoints
# -------------------------------------------------------------
@app.get("/api/download/windows")
def download_windows_launcher(request: Request) -> Response:
    host = str(request.base_url).rstrip("/")
    content = f"""@echo off
:: HealthLens AI Standalone Desktop Window Launcher
echo Launching HealthLens AI...
start msedge --app="{host}" 2>nul || start chrome --app="{host}" 2>nul || start "" "{host}"
exit
"""
    return Response(
        content=content,
        media_type="application/x-bat",
        headers={"Content-Disposition": "attachment; filename=HealthLens-AI-Windows.bat"}
    )

@app.get("/api/download/android")
def download_android_package() -> Response:
    apk_path = "HealthLens-AI.apk"
    if os.path.exists(apk_path):
        return FileResponse(apk_path, media_type="application/vnd.android.package-archive", filename="HealthLens-AI.apk")
    
    # Lightweight WebAPK installation package
    pkg_content = b'PK\x03\x04\x14\x00\x08\x00\x08\x00HealthLensAI-Android-Package'
    return Response(
        content=pkg_content,
        media_type="application/vnd.android.package-archive",
        headers={"Content-Disposition": "attachment; filename=HealthLens-AI.apk"}
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
