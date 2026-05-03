# NSE intraday trading signal app

Personal dashboard that ranks liquid NSE names by volume, computes RSI, MACD, volume vs 20-day average, price vs 20-day MA, pulls same-day news from NewsAPI, asks **Anthropic Claude** (`claude-sonnet-4-5`) to pick **one** intraday idea, stores it in **Supabase**, and optionally resolves outcomes after market close via `result_checker.py`.

## Project layout

| File | Purpose |
|------|---------|
| `analysis_engine.py` | Data + Claude + insert into `daily_signals` |
| `result_checker.py` | After 3:30 PM IST, mark WIN / LOSS / EXPIRED for pending rows |
| `app.py` | Flask API + serves `index.html` |
| `index.html` | Dark dashboard UI |
| `requirements.txt` | Python dependencies |
| `.env` | Secrets (not committed) |

## Environment variables

Create `.env` in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key
NEWSAPI_KEY=your_newsapi_key
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your_supabase_key
```

Use a Supabase **service role** key (or Row Level Security policies that allow insert/update for `daily_signals`) so the server can write rows.

## Supabase schema

Run this in the Supabase SQL editor:

```sql
create table if not exists public.daily_signals (
  id uuid primary key default gen_random_uuid(),
  signal_date date not null,
  ticker text not null,
  direction text not null check (direction in ('BUY', 'SELL')),
  entry_price numeric not null,
  target_price numeric not null,
  stop_loss numeric not null,
  confidence_score numeric not null,
  reasoning text,
  result text not null default 'PENDING'
    check (result in ('WIN', 'LOSS', 'EXPIRED', 'PENDING')),
  acted boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists daily_signals_signal_date_idx
  on public.daily_signals (signal_date desc, created_at desc);
```

## Setup

```bash
cd /path/to/financial-app
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the web app

```bash
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) (default port **8080**; override with `PORT`).

- **Generate today’s signal** calls `POST /generate-signal` (runs `analysis_engine.run_analysis()`).
- The dashboard loads **`GET /signals`** for the last 7 days and toggles **`POST /update-acted`** for the “I acted” flag.

## Run the end-of-day result checker

Runs a loop and **once per IST calendar day after 15:30**, evaluates all rows with `result = PENDING` using that session’s high/low from Yahoo Finance.

```bash
python result_checker.py
```

Keep this running on a machine that has network access to Yahoo Finance and Supabase, or schedule it with **cron** / a process manager instead of the built-in loop.

### Outcome rules (daily OHLC)

- **BUY:** `LOSS` if low ≤ stop_loss; else `WIN` if high ≥ target; else `EXPIRED`.
- **SELL:** `LOSS` if high ≥ stop_loss; else `WIN` if low ≤ target; else `EXPIRED`.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `index.html` |
| `POST` | `/generate-signal` | Runs analysis and inserts today’s signal |
| `GET` | `/signals` | Last 7 days of rows + win rate + today’s row |
| `POST` | `/update-acted` | Body: `{"id": "<uuid>", "acted": true}` |

## Notes

- **Not financial advice.** This is automation for personal experimentation.
- Yahoo Finance / NewsAPI / Claude availability may vary; `analysis_engine` skips tickers that fail metrics and aborts if none succeed.
- Ensure your Anthropic project allows model id **`claude-sonnet-4-5`** (adjust `ANTHROPIC_MODEL` in `analysis_engine.py` if your account uses a different snapshot string).
