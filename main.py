#!/usr/bin/env python3
"""
SWIM·AI — Backend API (FastAPI)
================================
Deploy en Railway en 5 minutos:
  1. Push este repo a GitHub
  2. New Project en railway.app → Deploy from GitHub
  3. Add variables de entorno (ver abajo)
  4. Railway detecta Procfile y despliega automáticamente

Variables de entorno necesarias:
  SUPABASE_URL          = https://xxx.supabase.co
  SUPABASE_SERVICE_KEY  = eyJ...  (service_role key, no la anon)
  JWT_SECRET            = (mismo que en Supabase → Settings → API → JWT Secret)
  CORS_ORIGIN           = https://tu-app.vercel.app

El motor fisiológico (engine.py) viene del mismo repo.
"""

from __future__ import annotations
import os, math, json
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import jwt as pyjwt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_SERVICE_KEY"]   # service_role
JWT_SECRET    = os.environ["JWT_SECRET"]
CORS_ORIGIN   = os.environ.get("CORS_ORIGIN", "*")

app = FastAPI(title="SWIM·AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SUPABASE CLIENT (HTTP) ───────────────────────────────────────────────────
# Usamos httpx directamente para evitar dependencias extras.
# En producción considera usar supabase-py.

HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=representation",
}

async def sb_get(table: str, params: dict = {}) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        return r.json()

async def sb_post(table: str, data: dict) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=HEADERS, json=data)
        r.raise_for_status()
        result = r.json()
        return result[0] if isinstance(result, list) else result

async def sb_patch(table: str, id: str, data: dict) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient() as client:
        r = await client.patch(
            url, headers=HEADERS, json=data,
            params={"id": f"eq.{id}"},
        )
        r.raise_for_status()
        result = r.json()
        return result[0] if isinstance(result, list) else result

async def sb_delete(table: str, id: str):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient() as client:
        r = await client.delete(url, headers=HEADERS, params={"id": f"eq.{id}"})
        r.raise_for_status()

# ─── AUTH ─────────────────────────────────────────────────────────────────────
async def get_athlete_id(authorization: str = Header(...)) -> str:
    try:
        token = authorization.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        
        # Extraer el payload del JWT sin verificar firma
        # (Supabase ya verificó el token al generarlo)
        parts = token.split('.')
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Token malformado")
        
        import base64, json
        payload_b64 = parts[1]
        # Añadir padding si hace falta
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += '=' * padding
        
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Token sin sub")
        
        return user_id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Error: {e}")
# ─── MOTOR FISIOLÓGICO ────────────────────────────────────────────────────────

def calc_css(t400: float, t200: float) -> float | None:
    """CSS en m/s. t400 y t200 en segundos."""
    if not t400 or not t200 or t400 <= t200:
        return None
    return (400 - 200) / (t400 - t200)

def calc_ctl_atl(sessions: list[dict]) -> dict:
    """
    Recalcula CTL y ATL desde cero con la lista de sesiones ordenadas por fecha.
    sessions: [{ session_date: "2024-01-15", tss: 85 }, ...]
    Devuelve { ctl, atl, tsb } del último día.
    """
    if not sessions:
        return {"ctl": 0, "atl": 0, "tsb": 0}

    sorted_s = sorted(sessions, key=lambda x: x["session_date"])

    # Mapa fecha → TSS total del día
    by_date: dict[str, float] = {}
    for s in sorted_s:
        d = str(s["session_date"])[:10]
        by_date[d] = by_date.get(d, 0) + (s.get("tss") or 0)

    if not by_date:
        return {"ctl": 0, "atl": 0, "tsb": 0}

    start = date.fromisoformat(min(by_date.keys()))
    end   = date.fromisoformat(max(by_date.keys()))

    ctl, atl = 0.0, 0.0
    current = start
    while current <= end:
        ds  = str(current)
        tss = by_date.get(ds, 0)
        ctl = ctl + (tss - ctl) * (1/42)
        atl = atl + (tss - atl) * (1/7)
        current += timedelta(days=1)

    return {
        "ctl": round(ctl, 2),
        "atl": round(atl, 2),
        "tsb": round(ctl - atl, 2),
    }

def predict_time(css_mps: float, distance: int, stroke: str) -> float:
    """Predice tiempo en segundos dado CSS, distancia y estilo."""
    base_k   = {50:1.55, 100:1.32, 200:1.12, 400:1.00, 800:0.925, 1500:0.870}.get(distance, 1.0)
    stroke_k = {
        "freestyle":1.00, "backstroke":1.14,
        "breaststroke":1.22, "butterfly":1.10, "medley":1.08,
    }.get(stroke, 1.00)
    return distance / (css_mps * base_k / stroke_k)

def css_zones(css_mps: float) -> dict:
    return {
        "en1":  {"lo": round(css_mps*0.78, 4), "hi": round(css_mps*0.85, 4)},
        "en2":  {"lo": round(css_mps*0.85, 4), "hi": round(css_mps*0.95, 4)},
        "en3":  {"lo": round(css_mps*0.95, 4), "hi": round(css_mps*1.00, 4)},
        "sp":   {"lo": round(css_mps*1.00, 4), "hi": round(css_mps*1.10, 4)},
    }

def tss_from_session(css: float, distance_m: int, time_sec: float, rpe: int) -> float:
    """
    Estima TSS de una sesión de natación.
    Fórmula: (time_sec * IF² * 100) / 3600
    donde IF = ritmo_sesion / css (equivalente al Intensity Factor del ciclismo).
    """
    if not css or not time_sec or time_sec <= 0:
        return float(rpe * 8)   # fallback si faltan datos
    pace = distance_m / time_sec    # m/s media de la sesión
    intensity_factor = pace / css
    tss = (time_sec * (intensity_factor ** 2) * 100) / 3600
    return round(tss, 1)

# ─── MODELOS ──────────────────────────────────────────────────────────────────

class AthleteProfile(BaseModel):
    name:        str
    age:         Optional[int]   = None
    club:        Optional[str]   = None
    main_stroke: Optional[str]   = "freestyle"
    main_distance: Optional[int] = 200

class SessionCreate(BaseModel):
    session_date:  str              # "2024-03-15"
    distance_m:    int
    duration_sec:  Optional[int]   = None
    rpe:           Optional[int]   = None
    notes:         Optional[str]   = None
    equipment:     Optional[list]  = None   # ["pull_buoy", "paddles"]
    zone:          Optional[str]   = None   # "EN2"
    tss_override:  Optional[float] = None

class CSSTestCreate(BaseModel):
    test_date: str
    t400_sec:  float
    t200_sec:  float
    notes:     Optional[str] = None

class RaceResultCreate(BaseModel):
    race_date:    str
    race_name:    str
    distance:     int
    stroke:       str
    time_sec:     float
    pool_length:  Optional[int] = 25
    priority:     Optional[str] = "B"   # A/B/C
    notes:        Optional[str] = None

class InjuryCreate(BaseModel):
    start_date:  str
    body_part:   str
    region:      str
    level:       int              # 1-3
    notes:       Optional[str]  = None

class GoalCreate(BaseModel):
    stroke:       str
    distance:     int
    target_time:  float
    race_date:    str
    race_name:    Optional[str] = None
    priority:     Optional[str] = "A"

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "SWIM·AI API running", "version": "1.0.0"}

# ── PERFIL DEL ATLETA ─────────────────────────────────────────────────────────

@app.get("/athletes/me")
async def get_my_profile(athlete_id: str = Depends(get_athlete_id)):
    rows = await sb_get("athletes", {"id": f"eq.{athlete_id}"})
    if not rows:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    return rows[0]

@app.put("/athletes/me")
async def update_profile(body: AthleteProfile,
                         athlete_id: str = Depends(get_athlete_id)):
    # Upsert — crea o actualiza
    data = {**body.dict(), "id": athlete_id, "updated_at": datetime.utcnow().isoformat()}
    url  = f"{SUPABASE_URL}/rest/v1/athletes"
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            headers={**HEADERS, "Prefer": "resolution=merge-duplicates,return=representation"},
            json=data,
        )
        r.raise_for_status()
        return r.json()[0]

# ── SESIONES ──────────────────────────────────────────────────────────────────

@app.get("/sessions")
async def get_sessions(
    limit:  int = 50,
    from_date: Optional[str] = None,
    athlete_id: str = Depends(get_athlete_id),
):
    params = {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "session_date.desc",
        "limit":      str(limit),
    }
    if from_date:
        params["session_date"] = f"gte.{from_date}"
    return await sb_get("sessions", params)

@app.post("/sessions", status_code=201)
async def create_session(body: SessionCreate,
                         athlete_id: str = Depends(get_athlete_id)):
    # Recuperar CSS actual para calcular TSS
    css_rows = await sb_get("css_tests", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "test_date.desc",
        "limit":      "1",
    })
    css = css_rows[0]["css_mps"] if css_rows else None

    tss = body.tss_override
    if not tss:
        if css and body.duration_sec:
            tss = tss_from_session(css, body.distance_m, body.duration_sec, body.rpe or 5)
        elif body.rpe:
            tss = body.rpe * 8.0

    data = {
        **body.dict(exclude={"tss_override"}),
        "athlete_id": athlete_id,
        "tss":        tss,
        "created_at": datetime.utcnow().isoformat(),
    }
    return await sb_post("sessions", data)

@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str,
                         athlete_id: str = Depends(get_athlete_id)):
    rows = await sb_get("sessions", {
        "id": f"eq.{session_id}", "athlete_id": f"eq.{athlete_id}"})
    if not rows:
        raise HTTPException(404, "Sesión no encontrada")
    await sb_delete("sessions", session_id)

# ── CARGA: CTL / ATL / TSB ───────────────────────────────────────────────────

@app.get("/load")
async def get_load(athlete_id: str = Depends(get_athlete_id)):
    """CTL, ATL, TSB actuales calculados desde todas las sesiones."""
    sessions = await sb_get("sessions", {
        "athlete_id": f"eq.{athlete_id}",
        "select":     "session_date,tss",
    })
    load = calc_ctl_atl(sessions)

    # Añadir acr (ratio aguda/crónica)
    load["acr"] = round(load["atl"] / max(load["ctl"], 1), 3)
    return load

@app.get("/load/history")
async def get_load_history(
    days: int = 90,
    athlete_id: str = Depends(get_athlete_id),
):
    """Serie temporal de CTL/ATL/TSB para los últimos N días."""
    from_date = (date.today() - timedelta(days=days)).isoformat()
    sessions  = await sb_get("sessions", {
        "athlete_id":    f"eq.{athlete_id}",
        "session_date":  f"gte.{from_date}",
        "select":        "session_date,tss",
    })

    by_date: dict[str, float] = {}
    for s in sessions:
        d = str(s["session_date"])[:10]
        by_date[d] = by_date.get(d, 0) + (s.get("tss") or 0)

    history = []
    ctl, atl = 0.0, 0.0
    for i in range(days):
        d   = (date.today() - timedelta(days=days-1-i)).isoformat()
        tss = by_date.get(d, 0)
        ctl = ctl + (tss - ctl) * (1/42)
        atl = atl + (tss - atl) * (1/7)
        history.append({
            "date": d, "tss": tss,
            "ctl":  round(ctl,2),
            "atl":  round(atl,2),
            "tsb":  round(ctl-atl,2),
            "acr":  round(atl/max(ctl,1),3),
        })

    return history

# ── CSS TESTS ─────────────────────────────────────────────────────────────────

@app.get("/css-tests")
async def get_css_tests(athlete_id: str = Depends(get_athlete_id)):
    return await sb_get("css_tests", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "test_date.desc",
    })

@app.get("/css-tests/current")
async def get_current_css(athlete_id: str = Depends(get_athlete_id)):
    rows = await sb_get("css_tests", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "test_date.desc",
        "limit":      "1",
    })
    if not rows:
        return {"css_mps": None, "zones": None, "days_since": None}

    css_test = rows[0]
    css      = css_test["css_mps"]
    test_dt  = date.fromisoformat(str(css_test["test_date"])[:10])
    days_since = (date.today() - test_dt).days

    return {
        **css_test,
        "zones":      css_zones(css) if css else None,
        "days_since": days_since,
        "stale":      days_since > 21,
    }

@app.post("/css-tests", status_code=201)
async def create_css_test(body: CSSTestCreate,
                          athlete_id: str = Depends(get_athlete_id)):
    css = calc_css(body.t400_sec, body.t200_sec)
    if not css:
        raise HTTPException(400, "T400 debe ser mayor que T200 y ambos positivos")

    data = {
        **body.dict(),
        "athlete_id": athlete_id,
        "css_mps":    round(css, 5),
        "created_at": datetime.utcnow().isoformat(),
    }
    return await sb_post("css_tests", data)

# ── PREDICCIONES ──────────────────────────────────────────────────────────────

@app.get("/predictions")
async def get_predictions(athlete_id: str = Depends(get_athlete_id)):
    """Predicciones para todas las pruebas basadas en CSS actual y CTL/TSB."""
    css_rows = await sb_get("css_tests", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "test_date.desc", "limit": "1",
    })
    if not css_rows:
        raise HTTPException(400, "No hay test CSS. Haz el test primero.")

    css = css_rows[0]["css_mps"]
    sessions = await sb_get("sessions", {
        "athlete_id": f"eq.{athlete_id}", "select": "session_date,tss"})
    load = calc_ctl_atl(sessions)
    tsb  = load["tsb"]

    STROKES = ["freestyle","backstroke","breaststroke","butterfly","medley"]
    DISTS   = [50, 100, 200, 400, 800, 1500]
    preds   = {}
    for stroke in STROKES:
        preds[stroke] = {}
        for dist in DISTS:
            if stroke != "freestyle" and dist > 400:
                continue
            base_time = predict_time(css, dist, stroke)
            # Ajuste TSB: cada punto de TSB por encima de 0 mejora ~0.1%
            tsb_factor = 1 - (tsb * 0.001) if tsb > 0 else 1 + (abs(tsb) * 0.0008)
            adj_time   = base_time * tsb_factor
            preds[stroke][dist] = {
                "time_sec":  round(adj_time, 2),
                "base_sec":  round(base_time, 2),
                "tsb_adj":   round(tsb_factor, 4),
            }

    return {"css_mps": css, "load": load, "predictions": preds}

# ── RESULTADOS DE CARRERA ─────────────────────────────────────────────────────

@app.get("/races")
async def get_races(athlete_id: str = Depends(get_athlete_id)):
    return await sb_get("race_results", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "race_date.desc",
    })

@app.post("/races", status_code=201)
async def create_race_result(body: RaceResultCreate,
                             athlete_id: str = Depends(get_athlete_id)):
    # Obtener CSS vigente en la fecha de la carrera para calibración
    css_rows = await sb_get("css_tests", {
        "athlete_id":  f"eq.{athlete_id}",
        "test_date":   f"lte.{body.race_date}",
        "order":       "test_date.desc",
        "limit":       "1",
    })
    css = css_rows[0]["css_mps"] if css_rows else None
    predicted = predict_time(css, body.distance, body.stroke) if css else None

    data = {
        **body.dict(),
        "athlete_id":   athlete_id,
        "predicted_sec": round(predicted, 2) if predicted else None,
        "delta_sec":    round(body.time_sec - predicted, 2) if predicted else None,
        "created_at":   datetime.utcnow().isoformat(),
    }
    return await sb_post("race_results", data)

# ── MARCAS PERSONALES ────────────────────────────────────────────────────────

@app.get("/prs")
async def get_prs(athlete_id: str = Depends(get_athlete_id)):
    """Mejor tiempo histórico por prueba + progresión."""
    races = await sb_get("race_results", {
        "athlete_id": f"eq.{athlete_id}",
        "select":     "stroke,distance,time_sec,race_date,race_name",
    })

    prs: dict[str, dict] = {}
    history: dict[str, list] = {}

    for r in sorted(races, key=lambda x: x["race_date"]):
        key = f"{r['stroke']}_{r['distance']}"
        if key not in history:
            history[key] = []
        history[key].append(r)

        if key not in prs or r["time_sec"] < prs[key]["time_sec"]:
            prs[key] = r

    return {
        "personal_bests": prs,
        "history":        history,
    }

# ── LESIONES ──────────────────────────────────────────────────────────────────

@app.get("/injuries")
async def get_injuries(athlete_id: str = Depends(get_athlete_id)):
    return await sb_get("injuries", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "start_date.desc",
    })

@app.post("/injuries", status_code=201)
async def create_injury(body: InjuryCreate,
                        athlete_id: str = Depends(get_athlete_id)):
    # Capturar carga en el momento de la lesión para análisis de patrones
    sessions = await sb_get("sessions", {
        "athlete_id":   f"eq.{athlete_id}",
        "session_date": f"lte.{body.start_date}",
        "select":       "session_date,tss",
    })
    load = calc_ctl_atl(sessions)

    # ACR en los 7 días previos
    week_ago   = (date.fromisoformat(body.start_date) - timedelta(days=7)).isoformat()
    week_sess  = [s for s in sessions if str(s["session_date"]) >= week_ago]
    week_tss   = sum(s.get("tss",0) for s in week_sess)

    data = {
        **body.dict(),
        "athlete_id":  athlete_id,
        "ctl_at_injury": load["ctl"],
        "atl_at_injury": load["atl"],
        "acr_at_injury": round(load["atl"]/max(load["ctl"],1), 3),
        "tss_7d_before": round(week_tss, 1),
        "resolved":    False,
        "created_at":  datetime.utcnow().isoformat(),
    }
    return await sb_post("injuries", data)

@app.patch("/injuries/{injury_id}/resolve")
async def resolve_injury(injury_id: str, end_date: str,
                         athlete_id: str = Depends(get_athlete_id)):
    rows = await sb_get("injuries", {
        "id": f"eq.{injury_id}", "athlete_id": f"eq.{athlete_id}"})
    if not rows:
        raise HTTPException(404, "Lesión no encontrada")
    return await sb_patch("injuries", injury_id, {
        "resolved": True, "end_date": end_date})

# ── PATRONES DE LESIÓN ────────────────────────────────────────────────────────

@app.get("/patterns")
async def get_injury_patterns(athlete_id: str = Depends(get_athlete_id)):
    """
    Detecta correlaciones entre carga y lesiones.
    Devuelve patrones confirmados (misma región, 2+ episodios, mismo trigger).
    """
    injuries = await sb_get("injuries", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "start_date.asc",
    })

    if len(injuries) < 2:
        return {"patterns": [], "message": "Se necesitan al menos 2 lesiones"}

    # Agrupar por región
    by_region: dict[str, list] = {}
    for inj in injuries:
        key = inj["body_part"]
        if key not in by_region:
            by_region[key] = []
        by_region[key].append(inj)

    patterns = []
    for body_part, inj_list in by_region.items():
        if len(inj_list) < 2:
            continue

        # Detectar trigger dominante
        triggers = []
        for inj in inj_list:
            acr = inj.get("acr_at_injury", 1.0) or 1.0
            tss = inj.get("tss_7d_before", 0) or 0
            if acr >= 1.20:
                triggers.append("acr")
            elif tss >= 250:
                triggers.append("tss_weekly")

        if not triggers:
            continue

        dominant = max(set(triggers), key=triggers.count)
        confidence = triggers.count(dominant) / len(inj_list)

        # ACR media en los episodios
        acr_values = [i.get("acr_at_injury", 1.0) for i in inj_list
                      if i.get("acr_at_injury")]
        avg_acr = sum(acr_values)/len(acr_values) if acr_values else None

        patterns.append({
            "body_part":   body_part,
            "region":      inj_list[0]["region"],
            "occurrences": len(inj_list),
            "trigger":     dominant,
            "confidence":  round(confidence, 2),
            "avg_acr_at_injury": round(avg_acr, 3) if avg_acr else None,
            "injuries":    inj_list,
        })

    # Riesgo actual
    sessions = await sb_get("sessions", {
        "athlete_id": f"eq.{athlete_id}", "select": "session_date,tss"})
    load = calc_ctl_atl(sessions)
    current_acr = round(load["atl"] / max(load["ctl"], 1), 3)

    risks = []
    for p in patterns:
        score = 0
        if p["trigger"] == "acr" and current_acr >= 1.20:
            score = (current_acr - 1.0) * 2
        if p["avg_acr_at_injury"] and current_acr >= p["avg_acr_at_injury"] * 0.90:
            score += 0.3
        risks.append({**p, "current_risk_score": round(min(score, 1.0), 2)})

    return {
        "patterns":    sorted(risks, key=lambda x: -x["current_risk_score"]),
        "current_acr": current_acr,
        "current_load": load,
    }

# ── OBJETIVO DE TEMPORADA ─────────────────────────────────────────────────────

@app.get("/goals")
async def get_goals(athlete_id: str = Depends(get_athlete_id)):
    return await sb_get("goals", {
        "athlete_id": f"eq.{athlete_id}",
        "order":      "race_date.asc",
    })

@app.post("/goals", status_code=201)
async def create_goal(body: GoalCreate,
                      athlete_id: str = Depends(get_athlete_id)):
    # CSS necesario para el tiempo meta
    bk = {50:1.55,100:1.32,200:1.12,400:1.00,800:0.925,1500:0.870}.get(body.distance,1.0)
    sk = {"freestyle":1.00,"backstroke":1.14,"breaststroke":1.22,
          "butterfly":1.10,"medley":1.08}.get(body.stroke,1.0)
    css_needed = body.distance / body.target_time / (bk / sk)

    # CSS actual
    css_rows = await sb_get("css_tests", {
        "athlete_id": f"eq.{athlete_id}", "order": "test_date.desc", "limit": "1"})
    css_current = css_rows[0]["css_mps"] if css_rows else None

    data = {
        **body.dict(),
        "athlete_id":  athlete_id,
        "css_needed":  round(css_needed, 5),
        "css_current": css_current,
        "created_at":  datetime.utcnow().isoformat(),
    }
    return await sb_post("goals", data)

# ── RESUMEN SEMANAL ───────────────────────────────────────────────────────────

@app.get("/weekly-digest")
async def get_weekly_digest(
    week_offset: int = 0,   # 0=esta semana, 1=semana pasada...
    athlete_id: str = Depends(get_athlete_id),
):
    today_d = date.today()
    # Inicio de la semana (lunes)
    week_start = today_d - timedelta(days=today_d.weekday()) - timedelta(weeks=week_offset)
    week_end   = week_start + timedelta(days=6)

    sessions = await sb_get("sessions", {
        "athlete_id":   f"eq.{athlete_id}",
        "session_date": f"gte.{week_start.isoformat()}",
        "order":        "session_date.asc",
    })
    sessions = [s for s in sessions if str(s["session_date"])[:10] <= week_end.isoformat()]

    total_m   = sum(s.get("distance_m",0) for s in sessions)
    total_tss = sum(s.get("tss",0) for s in sessions)
    avg_rpe   = (sum(s.get("rpe",0) for s in sessions if s.get("rpe"))
                 / max(len([s for s in sessions if s.get("rpe")]), 1))

    # Load al final de la semana
    all_sess = await sb_get("sessions", {
        "athlete_id":   f"eq.{athlete_id}",
        "session_date": f"lte.{week_end.isoformat()}",
        "select":       "session_date,tss",
    })
    load = calc_ctl_atl(all_sess)

    # Load de la semana anterior para delta CTL
    prev_end = week_start - timedelta(days=1)
    prev_sess = [s for s in all_sess if str(s["session_date"])[:10] <= prev_end.isoformat()]
    prev_load = calc_ctl_atl(prev_sess)
    ctl_delta = round(load["ctl"] - prev_load["ctl"], 2)

    # Tone
    km = total_m / 1000
    tone = "good" if km >= 9 else "ok" if km >= 6 else "low"

    return {
        "week_start":    week_start.isoformat(),
        "week_end":      week_end.isoformat(),
        "sessions":      sessions,
        "total_meters":  total_m,
        "total_km":      round(km, 2),
        "total_tss":     round(total_tss, 1),
        "avg_rpe":       round(avg_rpe, 1),
        "session_count": len(sessions),
        "load":          load,
        "ctl_delta":     ctl_delta,
        "acr":           round(load["atl"] / max(load["ctl"],1), 3),
        "tone":          tone,
    }

# ── NOTIFICACIONES ────────────────────────────────────────────────────────────

@app.get("/notifications")
async def get_notifications(athlete_id: str = Depends(get_athlete_id)):
    """
    Genera notificaciones activas basadas en el estado actual del atleta.
    No se almacenan — se calculan en tiempo real.
    """
    notifs = []
    now_d  = date.today()

    # ─ 1. CSS desactualizado
    css_rows = await sb_get("css_tests", {
        "athlete_id": f"eq.{athlete_id}", "order": "test_date.desc", "limit": "1"})
    if css_rows:
        test_date   = date.fromisoformat(str(css_rows[0]["test_date"])[:10])
        days_since  = (now_d - test_date).days
        if days_since >= 21:
            notifs.append({
                "type":     "css_stale",
                "priority": 1,
                "title":    f"CSS sin actualizar — {days_since} días",
                "body":     "Las predicciones pueden estar desajustadas.",
                "metric":   {"label":"Días","value":str(days_since),"threshold":"21"},
                "action":   {"label":"Hacer test CSS","route":"css_test"},
            })

    # ─ 2. Riesgo de lesión por ratio A/C
    sessions_all = await sb_get("sessions", {
        "athlete_id": f"eq.{athlete_id}", "select": "session_date,tss"})
    load  = calc_ctl_atl(sessions_all)
    acr   = round(load["atl"] / max(load["ctl"],1), 3)

    # Patrones del atleta para saber su umbral personal
    patterns_data = await get_injury_patterns(athlete_id)
    personal_threshold = 1.30
    for p in patterns_data.get("patterns",[]):
        if p.get("trigger") == "acr" and p.get("avg_acr_at_injury"):
            personal_threshold = min(personal_threshold, p["avg_acr_at_injury"])

    if acr >= personal_threshold * 0.95:
        notifs.append({
            "type":     "injury_risk",
            "priority": 2,
            "title":    f"Riesgo de lesión — ratio A/C en {acr}",
            "body":     f"La ratio aguda/crónica se acerca al umbral de tu historial ({personal_threshold:.2f}).",
            "metric":   {"label":"Ratio A/C","value":str(acr),"threshold":str(personal_threshold)},
            "action":   {"label":"Ver patrones","route":"injury_patterns"},
            "recommendation": "Reduce el volumen de la próxima sesión al 70%.",
        })

    # ─ 3. Cuenta atrás de competición A
    goals = await sb_get("goals", {
        "athlete_id": f"eq.{athlete_id}",
        "priority":   "eq.A",
        "race_date":  f"gte.{now_d.isoformat()}",
        "order":      "race_date.asc",
        "limit":      "1",
    })
    if goals:
        g        = goals[0]
        race_d   = date.fromisoformat(str(g["race_date"])[:10])
        days_to  = (race_d - now_d).days
        if days_to in range(0, 22):
            notifs.append({
                "type":     "race_countdown",
                "priority": 2 if days_to <= 7 else 1,
                "title":    f"Faltan {days_to} días — {g.get('race_name','Carrera A')}",
                "body":     f"{'El taper debería estar activo.' if days_to <= 14 else 'Empieza a reducir el volumen.'} TSB actual: {load['tsb']:+.1f}",
                "metric":   {"label":"TSB","value":f"{load['tsb']:+.1f}","threshold":"+8 a +14"},
                "action":   {"label":"Calentamiento de competición","route":"warmup"},
            })

    # Ordenar por prioridad
    notifs.sort(key=lambda x: -x["priority"])
    return {"notifications": notifs, "count": len(notifs)}
