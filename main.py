from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, List, Optional

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# Carica variabili d'ambiente da .env (se presente)
load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL non impostata. Imposta la variabile d'ambiente, "
        "ad esempio: postgresql://user:password@localhost:5432/trading_db",
    )

# ======== CONFIG DASH (env) ========
# Capitale iniziale desiderato (default 19$)
STARTING_EQUITY_USD = float(os.getenv("STARTING_EQUITY_USD", "19"))

# Data/ora da cui far partire performance e dati dashboard (ISO 8601)
# Esempio: 2025-12-12T13:00:00+01:00
DASH_START_AT_RAW = os.getenv("DASH_START_AT")
try:
    DASH_START_AT: Optional[datetime] = (
        datetime.fromisoformat(DASH_START_AT_RAW) if DASH_START_AT_RAW else None
    )
except ValueError:
    # Se formato sbagliato, non blocchiamo la dashboard: semplicemente non filtriamo
    DASH_START_AT = None


@contextmanager
def get_connection():
    """Context manager che restituisce una connessione PostgreSQL.

    Usa il DSN in DATABASE_URL.
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


# =====================
# Modelli di risposta API
# =====================


class BalancePoint(BaseModel):
    timestamp: datetime
    balance_usd: float


class OpenPosition(BaseModel):
    id: int
    snapshot_id: int
    symbol: str
    side: str
    size: float
    entry_price: Optional[float]
    mark_price: Optional[float]
    pnl_usd: Optional[float]
    leverage: Optional[str]
    snapshot_created_at: datetime


class ClosedPosition(BaseModel):
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    opened_at: datetime
    closed_at: datetime
    leverage: Optional[str]


class BotOperation(BaseModel):
    id: int
    created_at: datetime
    operation: str
    symbol: Optional[str]
    direction: Optional[str]
    target_portion_of_balance: Optional[float]
    leverage: Optional[float]
    raw_payload: Any
    system_prompt: Optional[str]
    # Dati tecnici aggiuntivi (da indicators_contexts e forecasts_contexts)
    rsi_7: Optional[float] = None
    macd: Optional[float] = None
    current_price: Optional[float] = None
    predicted_price: Optional[float] = None
    forecast_lower: Optional[float] = None
    forecast_upper: Optional[float] = None


# =====================
# App FastAPI + Template Jinja2
# =====================


app = FastAPI(
    title="Trading Agent Dashboard API",
    description=(
        "API per leggere i dati del trading agent dal database Postgres: "
        "saldo nel tempo, posizioni aperte, operazioni del bot con full prompt."
    ),
    version="0.3.2",
)

templates = Jinja2Templates(directory="templates")


# =====================
# Endpoint API JSON
# =====================


@app.get("/balance", response_model=List[BalancePoint])
def get_balance() -> List[BalancePoint]:
    """Restituisce la storia del saldo (balance_usd) ordinata nel tempo.

    Se DASH_START_AT è impostata, filtra i punti a partire da quella data/ora.
    I dati sono presi dalla tabella `account_snapshots`.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if DASH_START_AT:
                cur.execute(
                    """
                    SELECT created_at, balance_usd
                    FROM account_snapshots
                    WHERE created_at >= %s
                    ORDER BY created_at ASC;
                    """,
                    (DASH_START_AT,),
                )
            else:
                cur.execute(
                    """
                    SELECT created_at, balance_usd
                    FROM account_snapshots
                    ORDER BY created_at ASC;
                    """
                )
            rows = cur.fetchall()

    return [BalancePoint(timestamp=row[0], balance_usd=float(row[1])) for row in rows]


@app.get("/open-positions", response_model=List[OpenPosition])
def get_open_positions() -> List[OpenPosition]:
    """Restituisce le posizioni aperte dell'ULTIMO snapshot disponibile.

    Se DASH_START_AT è impostata, prende l'ultimo snapshot a partire da quella data/ora.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Ultimo snapshot (filtrato se richiesto)
            if DASH_START_AT:
                cur.execute(
                    """
                    SELECT id, created_at
                    FROM account_snapshots
                    WHERE created_at >= %s
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """,
                    (DASH_START_AT,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, created_at
                    FROM account_snapshots
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """
                )

            row = cur.fetchone()
            if not row:
                return []
            snapshot_id = row[0]
            snapshot_created_at = row[1]

            # Posizioni aperte per quello snapshot
            cur.execute(
                """
                SELECT
                    id,
                    snapshot_id,
                    symbol,
                    side,
                    size,
                    entry_price,
                    mark_price,
                    pnl_usd,
                    leverage
                FROM open_positions
                WHERE snapshot_id = %s
                ORDER BY symbol ASC, id ASC;
                """,
                (snapshot_id,),
            )
            rows = cur.fetchall()

    return [
        OpenPosition(
            id=row[0],
            snapshot_id=row[1],
            symbol=row[2],
            side=row[3],
            size=float(row[4]),
            entry_price=float(row[5]) if row[5] is not None else None,
            mark_price=float(row[6]) if row[6] is not None else None,
            pnl_usd=float(row[7]) if row[7] is not None else None,
            leverage=row[8],
            snapshot_created_at=snapshot_created_at,
        )
        for row in rows
    ]


@app.get("/closed-positions", response_model=List[ClosedPosition])
def get_closed_positions() -> List[ClosedPosition]:
    """Calcola le posizioni chiuse confrontando gli snapshot consecutivi.

    Se DASH_START_AT è impostata, usa solo gli snapshot a partire da quella data/ora.
    Logica:
    - Se una posizione esiste in T ma non in T+1, è stata chiusa.
    - Il PnL realizzato è l'ultimo PnL registrato in T.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if DASH_START_AT:
                cur.execute(
                    """
                    SELECT 
                        s.id, 
                        s.created_at,
                        op.symbol,
                        op.side,
                        op.entry_price,
                        op.mark_price,
                        op.pnl_usd,
                        op.leverage
                    FROM account_snapshots s
                    JOIN open_positions op ON s.id = op.snapshot_id
                    WHERE s.created_at >= %s
                    ORDER BY s.created_at ASC;
                    """,
                    (DASH_START_AT,),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        s.id, 
                        s.created_at,
                        op.symbol,
                        op.side,
                        op.entry_price,
                        op.mark_price,
                        op.pnl_usd,
                        op.leverage
                    FROM account_snapshots s
                    JOIN open_positions op ON s.id = op.snapshot_id
                    ORDER BY s.created_at ASC;
                    """
                )
            rows = cur.fetchall()

    snapshots_map = {}
    for row in rows:
        snap_id = row[0]
        created_at = row[1]

        if snap_id not in snapshots_map:
            snapshots_map[snap_id] = {"created_at": created_at, "positions": {}}

        pos_key = f"{row[2]}_{row[3]}"
        snapshots_map[snap_id]["positions"][pos_key] = {
            "symbol": row[2],
            "side": row[3],
            "entry_price": float(row[4]) if row[4] is not None else 0,
            "mark_price": float(row[5]) if row[5] is not None else 0,
            "pnl_usd": float(row[6]) if row[6] is not None else 0,
            "leverage": row[7],
        }

    sorted_snap_ids = sorted(
        snapshots_map.keys(), key=lambda k: snapshots_map[k]["created_at"]
    )

    closed_positions = []
    position_start_times = {}

    for i in range(len(sorted_snap_ids)):
        curr_id = sorted_snap_ids[i]
        curr_snap = snapshots_map[curr_id]
        curr_positions = curr_snap["positions"]
        curr_time = curr_snap["created_at"]

        for pos_key in curr_positions:
            if pos_key not in position_start_times:
                position_start_times[pos_key] = curr_time

        if i < len(sorted_snap_ids) - 1:
            next_id = sorted_snap_ids[i + 1]
            next_snap = snapshots_map[next_id]
            next_positions = next_snap["positions"]
            next_time = next_snap["created_at"]

            for pos_key, pos_data in curr_positions.items():
                if pos_key not in next_positions:
                    opened_at = position_start_times.get(pos_key, curr_time)

                    closed_positions.append(
                        ClosedPosition(
                            symbol=pos_data["symbol"],
                            side=pos_data["side"],
                            entry_price=pos_data["entry_price"],
                            exit_price=pos_data["mark_price"],
                            pnl_usd=pos_data["pnl_usd"],
                            opened_at=opened_at,
                            closed_at=next_time,
                            leverage=pos_data["leverage"],
                        )
                    )

                    if pos_key in position_start_times:
                        del position_start_times[pos_key]

    closed_positions.sort(key=lambda x: x.closed_at, reverse=True)
    return closed_positions


@app.get("/bot-operations", response_model=List[BotOperation])
def get_bot_operations(
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Numero massimo di operazioni da restituire (default 50)",
    ),
) -> List[BotOperation]:
    """Restituisce le ULTIME `limit` operazioni del bot con il full system prompt.

    Nota: le operazioni sono ordinate per created_at DESC.
    Se vuoi anche qui filtrare per DASH_START_AT, aggiungiamo WHERE bo.created_at >= %s.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if DASH_START_AT:
                cur.execute(
                    """
                    SELECT
                        bo.id,
                        bo.created_at,
                        bo.operation,
                        bo.symbol,
                        bo.direction,
                        bo.target_portion_of_balance,
                        bo.leverage,
                        bo.raw_payload,
                        ac.system_prompt,
                        ic.rsi_7,
                        ic.macd,
                        ic.price,
                        fc.prediction,
                        fc.lower_bound,
                        fc.upper_bound
                    FROM bot_operations AS bo
                    LEFT JOIN ai_contexts AS ac ON bo.context_id = ac.id
                    LEFT JOIN indicators_contexts AS ic ON bo.context_id = ic.context_id AND bo.symbol = ic.ticker
                    LEFT JOIN forecasts_contexts AS fc ON bo.context_id = fc.context_id AND bo.symbol = fc.ticker
                    WHERE bo.created_at >= %s
                    ORDER BY bo.created_at DESC
                    LIMIT %s;
                    """,
                    (DASH_START_AT, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        bo.id,
                        bo.created_at,
                        bo.operation,
                        bo.symbol,
                        bo.direction,
                        bo.target_portion_of_balance,
                        bo.leverage,
                        bo.raw_payload,
                        ac.system_prompt,
                        ic.rsi_7,
                        ic.macd,
                        ic.price,
                        fc.prediction,
                        fc.lower_bound,
                        fc.upper_bound
                    FROM bot_operations AS bo
                    LEFT JOIN ai_contexts AS ac ON bo.context_id = ac.id
                    LEFT JOIN indicators_contexts AS ic ON bo.context_id = ic.context_id AND bo.symbol = ic.ticker
                    LEFT JOIN forecasts_contexts AS fc ON bo.context_id = fc.context_id AND bo.symbol = fc.ticker
                    ORDER BY bo.created_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
            rows = cur.fetchall()

    operations: List[BotOperation] = []
    for row in rows:
        operations.append(
            BotOperation(
                id=row[0],
                created_at=row[1],
                operation=row[2],
                symbol=row[3],
                direction=row[4],
                target_portion_of_balance=float(row[5]) if row[5] is not None else None,
                leverage=float(row[6]) if row[6] is not None else None,
                raw_payload=row[7],
                system_prompt=row[8],
                rsi_7=float(row[9]) if row[9] is not None else None,
                macd=float(row[10]) if row[10] is not None else None,
                current_price=float(row[11]) if row[11] is not None else None,
                predicted_price=float(row[12]) if row[12] is not None else None,
                forecast_lower=float(row[13]) if row[13] is not None else None,
                forecast_upper=float(row[14]) if row[14] is not None else None,
            )
        )

    return operations


# =====================
# Endpoint HTML + HTMX
# =====================


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Dashboard principale HTML."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/ui/balance", response_class=HTMLResponse)
async def ui_balance(request: Request) -> HTMLResponse:
    """Partial HTML con il grafico del saldo nel tempo."""
    points = get_balance()
    labels = [p.timestamp.isoformat() for p in points]
    values = [p.balance_usd for p in points]
    return templates.TemplateResponse(
        "partials/balance_table.html",
        {"request": request, "labels": labels, "values": values},
    )


@app.get("/ui/open-positions", response_class=HTMLResponse)
async def ui_open_positions(request: Request) -> HTMLResponse:
    """Partial HTML con le posizioni aperte (ultimo snapshot)."""
    positions = get_open_positions()
    return templates.TemplateResponse(
        "partials/open_positions_table.html",
        {"request": request, "positions": positions},
    )


@app.get("/ui/bot-operations", response_class=HTMLResponse)
async def ui_bot_operations(request: Request) -> HTMLResponse:
    """Partial HTML con le ultime operazioni del bot."""
    operations = get_bot_operations(limit=10)
    return templates.TemplateResponse(
        "partials/bot_operations_table.html",
        {"request": request, "operations": operations},
    )


@app.get("/ui/closed-positions", response_class=HTMLResponse)
async def ui_closed_positions(request: Request) -> HTMLResponse:
    """Partial HTML con lo storico delle posizioni chiuse e statistiche (periodo filtrato)."""
    positions = get_closed_positions()

    total = len(positions)
    wins = len([p for p in positions if p.pnl_usd > 0])
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0

    stats = {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
    }

    # Mostriamo le ultime 20
    return templates.TemplateResponse(
        "partials/closed_positions_table.html",
        {
            "request": request,
            "positions": positions[:20],
            "stats": stats,
        },
    )


@app.get("/ui/pnl-stats", response_class=HTMLResponse)
async def ui_pnl_stats(request: Request) -> HTMLResponse:
    """Partial HTML con le statistiche di PnL (infografica) basate su STARTING_EQUITY_USD."""
    points = get_balance()

    # Se non ci sono punti (o non ci sono ancora punti dopo DASH_START_AT)
    if not points:
        current_stats = {
            "initial_balance": STARTING_EQUITY_USD,
            "current_balance": STARTING_EQUITY_USD,
            "pnl_usd": 0.0,
            "pnl_percent": 0.0,
        }
        return templates.TemplateResponse(
            "partials/pnl_stats.html",
            {
                "request": request,
                "current_stats": current_stats,
                "archive_stats": None,
                "has_data": True,
            },
        )

    initial_balance = STARTING_EQUITY_USD
    current_balance = points[-1].balance_usd
    pnl_usd = current_balance - initial_balance
    pnl_percent = (pnl_usd / initial_balance * 100) if initial_balance != 0 else 0

    current_stats = {
        "initial_balance": initial_balance,
        "current_balance": current_balance,
        "pnl_usd": pnl_usd,
        "pnl_percent": pnl_percent,
    }

    return templates.TemplateResponse(
        "partials/pnl_stats.html",
        {
            "request": request,
            "current_stats": current_stats,
            "archive_stats": None,
            "has_data": True,
        },
    )


# Comodo per sviluppo locale: `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
