# Enterprise AI Support Platform — Session 4: Persistence & Threading

Session 4 of 12 in a structured build of a production-grade AI support agent using LangGraph + Gemini 2.5 Flash. This session adds SQLite-backed conversation persistence and multi-thread isolation, making the agent stateful across requests and process restarts.

---

## What's new in Session 4

| Feature | Detail |
|---|---|
| SQLite checkpointer | Every graph step is saved to `support.db` via `SqliteSaver` |
| Thread isolation | Each conversation gets a `thread_id` (UUID); state never bleeds between users |
| Follow-up turns | Submitting a second message on the same thread resumes prior context automatically |
| Conversation history API | `GET /api/history/{thread_id}` returns every checkpoint for a thread |
| Thread list API | `GET /api/threads` lists all thread IDs that have checkpoints |
| Thread Selector UI | Dropdown to switch between conversations; history panel shows every graph step |
| Metric cards | Live Category / Tools / Iterations / SQLite cards update after each run |
| No double-execution | Stream and Run share one `thread_id`; `/api/run` returns the existing checkpoint instead of re-running |

---

## Project structure

```
session4/
├── support_agent.py   # LangGraph agent — graph, nodes, tools, persistence
├── api.py             # FastAPI server — REST endpoints + SSE stream
├── index.html         # Single-file frontend
├── verify_s4.js       # Playwright end-to-end verification suite (30 checks)
├── requirements.txt   # Python dependencies
└── support.db         # SQLite database (auto-created on first run)
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+ (only needed to run the Playwright verification suite)
- A Google AI API key with Gemini 2.5 Flash access

---

## Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Key packages and why they're here:

| Package | Purpose |
|---|---|
| `langgraph==1.2.0` | Graph execution engine |
| `langgraph-checkpoint-sqlite` | SQLite checkpointer for persistence |
| `langchain-google-genai` | Gemini 2.5 Flash LLM binding |
| `fastapi` + `uvicorn` | HTTP server and SSE streaming |
| `python-dotenv` | Load `GOOGLE_API_KEY` from `.env` |

---

## Step 2 — Set your API key

Create a `.env` file in the project directory:

```bash
echo "GOOGLE_API_KEY=your-key-here" > .env
```

Or export it in your shell:

```bash
export GOOGLE_API_KEY="your-key-here"
```

---

## Step 3 — Start the server

```bash
python api.py
```

You should see startup output like:

```
[System] Gemini 2.5 Flash initialized | temperature=0
[ReAct] MAX_ITERATIONS=5 | CONTEXT_THRESHOLD=12
[Checkpointer] SQLite initialized → support.db
[System] SupportState schema — 17 fields across 12 sessions
[Tools] 3 tools registered:
  · get_customer_details
  · search_knowledge_base
  · check_fraud_signals
[Graph] Session 4 — 8 nodes | SQLite persistence active
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The `support.db` file is created automatically on first startup. It persists across restarts — all previous conversations are reloaded.

---

## Step 4 — Open the UI

Navigate to `http://localhost:8000` in your browser.

### UI layout

```
┌─────────────────────────────┬──────────────────────────────┐
│  Left panel                 │  Right panel                 │
│  ┌─────────────────────┐   │  Thread Selector             │
│  │ 📂 — 🔧 — 🔁 — 💾 │   │  Conversation History        │
│  └─────────────────────┘   │  Iteration Tracker           │
│  Sample pills               │  Circuit Breaker             │
│  Ticket textarea            ├──────────────────────────────┤
│  Submit button              │  Live stream trace           │
│                             │  Result card                 │
│                             │  Tool call inspector         │
└─────────────────────────────┴──────────────────────────────┘
```

### Metric cards (top of left panel)

| Card | Shows |
|---|---|
| 📂 CATEGORY | Classification result — technical / billing / fraud / general |
| 🔧 TOOLS | Number of tool calls made in the last run |
| 🔁 ITERATIONS | ReAct loop count (max 5) |
| 💾 PERSISTENCE | Always "SQLite" — confirms checkpointer is active |

---

## Step 5 — Submit a ticket (first turn)

1. Leave the Thread Selector empty (new conversation).
2. Click the **💳 Double Charge** sample pill or type: `My account C-1002 is past due, please check`
3. Click **Submit Ticket →**

What happens internally:

```
1. JS sends POST /api/stream  { ticket, thread_id: null }
2. Server generates thread_id UUID, emits SSE start event: { type: "start", thread_id: "abc-123" }
3. JS captures streamThreadId from start event
4. Graph runs: classify_node → agent_node → tool_node (get_customer_details) → agent_node → respond_node
5. Each node emits an SSE event; trace chips animate in real time
6. JS sends POST /api/run  { ticket, thread_id: "abc-123", return_existing: true }
7. Backend finds the completed checkpoint, returns its state — graph does NOT run again
8. Thread ID is added to the selector dropdown
9. Conversation History panel populates with every graph checkpoint
10. 💾 Saved indicator flashes
```

---

## Step 6 — Continue the conversation (follow-up turn)

With the thread still selected in the dropdown, type: `What was the outstanding balance?`

The agent answers "$998" without calling `get_customer_details` again — it reads the prior CRM result directly from the SQLite checkpoint.

What happens internally:

```
1. JS reads thread_id from the selector — same UUID as before
2. POST /api/stream  { ticket: "What was...", thread_id: "abc-123" }
3. stream_ticket() calls graph.get_state_history(config) → len > 0 → is_first_turn = False
4. Graph resumes from checkpoint: sends only { messages: [HumanMessage("What was...")] }
5. Prior context (C-1002, $998, Arjun Mehta) is already in the checkpoint state
6. Agent responds in one iteration — no tool calls needed
7. History count grows in the panel
```

---

## Step 7 — Start a new conversation

Click **+ New Thread** in the Thread Selector panel.

- The selector clears to "New conversation"
- The next submit creates a brand-new `thread_id`
- The old thread remains in the dropdown and in `support.db`

To return to an earlier conversation, select its thread ID from the dropdown and submit a follow-up message.

---

## Step 8 — Run the built-in verification test

Click the **Verification** toggle in the right panel, then click **Run Session 4 Verification**.

The backend runs 5 automated checks:

| Check | What it proves |
|---|---|
| 1. Turn 1 creates checkpoints | `support.db` is being written |
| 2. Turn 2 loads prior context | Follow-up turns resume correctly |
| 3. Thread isolation | Two thread IDs never share state |
| 4. get_state_history() structured | Checkpoint ledger is queryable |
| 5. get_active_threads() works | All thread IDs are discoverable |

All 5 passing means Session 5 is unblocked.

---

## Step 9 — Run the CLI test harness

You can run all Session 4 tests from the terminal without the browser:

```bash
python support_agent.py
```

This runs 5 tests then calls `run_session_verification()` and prints a full pass/fail report.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/stream` | SSE stream — emits node events as the graph runs |
| `POST` | `/api/run` | Full run — returns complete result with tool call log |
| `GET` | `/api/threads` | Lists all thread IDs that have checkpoints |
| `GET` | `/api/history/{thread_id}` | Returns all checkpoints for a thread |
| `POST` | `/api/verify` | Runs the 5-check session verification test |
| `GET` | `/health` | Server status, session number, persistence info |

### Example: inspect a thread's history

```bash
# List all threads
curl http://localhost:8000/api/threads

# Get full checkpoint history for a thread
curl http://localhost:8000/api/history/YOUR-THREAD-UUID
```

---

## Key implementation details

### Why `SqliteSaver` is initialized directly (not via `from_conn_string`)

`from_conn_string` is a Python context manager — it yields the saver and then closes the connection when the `with` block exits. At module level there is no `with` block, so the connection closes immediately. The correct pattern:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = 'support.db'
_db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_db_conn)
```

`check_same_thread=False` is required because FastAPI serves requests across multiple threads.

### First-turn vs. follow-up detection

```python
config   = {'configurable': {'thread_id': thread_id}}
existing = list(graph.get_state_history(config))

if len(existing) == 0:
    # First turn — build full initial state
    state = build_initial_state(ticket)
else:
    # Follow-up — append only the new message; prior state loads from checkpoint
    state = {'messages': [HumanMessage(content=ticket)]}

graph.invoke(state, config=config)
```

### Preventing double-execution

The UI pattern streams the graph first (for live trace), then calls `/api/run` to get the structured result. Without the fix, this would run the graph twice — creating two separate threads.

The fix:
1. `/api/stream` generates the `thread_id` UUID before calling `stream_ticket` and emits it in the first SSE event: `{"type": "start", "thread_id": "abc-123"}`
2. JS captures `streamThreadId` from that event
3. `/api/run` receives `{ thread_id: streamThreadId, return_existing: true }`
4. `run_ticket` checks if the thread already ran to END; if so, returns the existing checkpoint state without invoking the graph

### Accessing checkpoint state

```python
# StateSnapshot attributes (LangGraph 1.2.0)
snap.values          # dict — the graph state fields
snap.next            # tuple of next node names; () means graph is at END
snap.metadata        # dict — step number, source node name
snap.config          # dict — contains checkpoint_id under configurable
```

---

## Graph architecture (Session 4)

```
                    ┌─────────────┐
                    │classify_node│  Gemini classifies → technical/billing/fraud/general
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      agent_node      fraud_handler   general_handler
     (technical &     (single tool    (stub — replies
      billing)         call + LLM)     in 24 hrs)
           │
     ┌─────┴──────┐
     ▼            ▼
 tool_node   respond_node
     │            │
     └────┐        └──► END
          ▼
      agent_node   (loops until no tool calls or MAX_ITERATIONS=5)
```

All edges and the `SqliteSaver` checkpointer are compiled once at module load (`graph = build_graph()`). The checkpointer intercepts every state transition and writes it to `support.db`.

---

## Running the Playwright verification suite

Requires Playwright with the Chromium browser:

```bash
npm install playwright
npx playwright install chromium
node verify_s4.js
```

Runs 30 automated checks covering every UI element, the full submit flow, follow-up context loading, thread selector behavior, `/api/threads`, and session progress indicators. All 30 must pass.

---

## Session series context

| Session | Topic | Status |
|---|---|---|
| 1 | The Blueprint — state schema, classifier, router | ✅ Complete |
| 2 | Tool Binding — CRM + KB tools, agent node | ✅ Complete |
| 3 | The ReAct Architecture — circuit breaker, duplicate detection | ✅ Complete |
| **4** | **Persistence & Threading — SQLite checkpointer, thread isolation** | **✅ Complete** |
| 5 | Context Management & Summarization | 🔜 Next |
| 6–12 | Safety, evaluation, multi-agent, HITL, production hardening | Planned |

Session 5 will add a `summarization_node` that compresses long conversation histories into a `system_summary` string before they hit the context window limit. The `SqliteSaver` checkpointer, `thread_id` patterns, and all Session 4 functions remain unchanged.
