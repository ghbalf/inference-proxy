import aiosqlite
import time
from pathlib import Path

DB_PATH = "./usage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    endpoint TEXT NOT NULL DEFAULT '',
    method TEXT NOT NULL DEFAULT 'POST',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    streaming INTEGER NOT NULL DEFAULT 0,
    status_code INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests(provider);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
"""


async def init_db(db_path: str | None = None):
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def log_request(
    db_path: str | None = None,
    *,
    provider: str,
    model: str = "",
    endpoint: str = "",
    method: str = "POST",
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
    duration_ms: float = 0.0,
    streaming: bool = False,
    status_code: int = 0,
    error: str = "",
):
    path = db_path or DB_PATH
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """INSERT INTO requests
               (timestamp, provider, model, endpoint, method,
                input_tokens, output_tokens, total_tokens, cost_usd,
                duration_ms, streaming, status_code, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                provider,
                model,
                endpoint,
                method,
                input_tokens,
                output_tokens,
                total_tokens,
                cost_usd,
                duration_ms,
                1 if streaming else 0,
                status_code,
                error,
            ),
        )
        await db.commit()


async def query_summary(db_path: str | None = None):
    path = db_path or DB_PATH
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        # Overall totals
        cur = await db.execute(
            """SELECT
                 COUNT(*) as total_requests,
                 COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                 COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                 COALESCE(SUM(total_tokens), 0) as total_tokens,
                 COALESCE(SUM(cost_usd), 0.0) as total_cost_usd
               FROM requests"""
        )
        totals = dict(await cur.fetchone())

        # Per-provider breakdown
        cur = await db.execute(
            """SELECT
                 provider,
                 COUNT(*) as requests,
                 COALESCE(SUM(input_tokens), 0) as input_tokens,
                 COALESCE(SUM(output_tokens), 0) as output_tokens,
                 COALESCE(SUM(total_tokens), 0) as total_tokens,
                 COALESCE(SUM(cost_usd), 0.0) as cost_usd
               FROM requests
               GROUP BY provider
               ORDER BY cost_usd DESC"""
        )
        providers = [dict(row) for row in await cur.fetchall()]

    return {"totals": totals, "by_provider": providers}


async def query_by_model(
    db_path: str | None = None, provider: str | None = None, days: int | None = None
):
    path = db_path or DB_PATH
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        conditions = []
        params = []
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if days:
            conditions.append("timestamp >= ?")
            params.append(time.time() - days * 86400)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        cur = await db.execute(
            f"""SELECT
                  provider, model,
                  COUNT(*) as requests,
                  COALESCE(SUM(input_tokens), 0) as input_tokens,
                  COALESCE(SUM(output_tokens), 0) as output_tokens,
                  COALESCE(SUM(total_tokens), 0) as total_tokens,
                  COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM requests
                {where}
                GROUP BY provider, model
                ORDER BY cost_usd DESC""",
            params,
        )
        return [dict(row) for row in await cur.fetchall()]


async def query_requests(
    db_path: str | None = None, limit: int = 50, offset: int = 0
):
    path = db_path or DB_PATH
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT * FROM requests ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        return [dict(row) for row in await cur.fetchall()]


async def query_timeseries(
    db_path: str | None = None, days: int = 30, bucket: str = "day"
):
    path = db_path or DB_PATH
    since = time.time() - days * 86400
    if bucket == "hour":
        fmt = "%Y-%m-%d %H:00"
    else:
        fmt = "%Y-%m-%d"

    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT
                  strftime('{fmt}', timestamp, 'unixepoch', 'localtime') as bucket,
                  COUNT(*) as requests,
                  COALESCE(SUM(input_tokens), 0) as input_tokens,
                  COALESCE(SUM(output_tokens), 0) as output_tokens,
                  COALESCE(SUM(total_tokens), 0) as total_tokens,
                  COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM requests
                WHERE timestamp >= ?
                GROUP BY bucket
                ORDER BY bucket ASC""",
            (since,),
        )
        return [dict(row) for row in await cur.fetchall()]
