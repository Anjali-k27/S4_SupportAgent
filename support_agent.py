"""
Enterprise AI Support Platform
Session 4 of 12 — Persistence & Threading

Extends Session 3 with SQLite-backed checkpointing.
State survives process restarts. Conversations are isolated
by thread_id. Time travel and HITL are now possible.

Run server: python api.py  → http://localhost:8000
Run CLI:    python support_agent.py
"""

import os
import time
import operator
import json
import uuid
import sqlite3
from typing import TypedDict, Annotated, Literal, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# ── Environment setup ────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY not set. Run: export OPENROUTER_API_KEY='your-key-here'"
    )

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_SITE_URL = os.environ.get(
    "OPENROUTER_SITE_URL",
    "https://github.com/Anjali-k27/S4_SupportAgent",
)
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "S4_SupportAgent")

llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    },
)
print(f"[System] OpenRouter initialized | model={OPENROUTER_MODEL} | temperature=0")

# ── ReAct Constants (Session 3) ─────────────────────────────────────────────
MAX_ITERATIONS    = 5
CONTEXT_THRESHOLD = 12

print(f"[ReAct] MAX_ITERATIONS={MAX_ITERATIONS} | "
      f"CONTEXT_THRESHOLD={CONTEXT_THRESHOLD}")

# ── Checkpointer (Session 4) ─────────────────────────────────────────────────

DB_PATH = 'support.db'

_db_conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_db_conn)

print(f"[Checkpointer] SQLite initialized → {DB_PATH}")


# ════════════════════════════════════════════════════════════════�[...]
# SECTION 2: STATE SCHEMA
# ════════════════════════════════════════════════════════════════�[...]

class SupportState(TypedDict):

    # ── Core Input (Session 1) ──────────────────────────────────
    raw_input:          str        # Original user message, never modified
    sanitized_input:    str        # PII-cleaned version (Session 6)

    # ── Classification (Session 1) ─────────────────────────────
    category:           str        # technical | billing | fraud | general

    # ── Conversation History (Session 2) ───────────────────────
    messages:           Annotated[list, add_messages]
    customer_data:      dict       # Populated by CRM tool
    tool_results:       Annotated[list, operator.add]  # Append-safe

    # ── Safety Controls (Session 6) ────────────────────────────
    pii_detected:       bool
    injection_detected: bool
    is_safe:            bool

    # ── Memory and Context (Sessions 3, 5) ─────────────────────
    system_summary:     str        # Compressed history (Session 5)
    iteration_count:    int        # ReAct circuit breaker (Session 3)

    # ── Multi-Agent Orchestration (Sessions 8, 9) ───────────────
    internal_notes:     Annotated[list, operator.add]  # Parallel scratchpad
    delegation_count:   int        # Supervisor counter
    next_worker:        str        # Supervisor decision

    # ── Write Access and Human Approval (Session 10, 11) ────────
    github_draft:       dict       # Proposed issue before approval
    github_issue_url:   str        # URL after creation

    # ── Output (Session 1) ─────────────────────────────────────
    final_response:     str


_field_count = len(SupportState.__annotations__)
print(f"[System] SupportState schema — {_field_count} fields across 12 sessions")


# ── Mock Data (Session 2) ──────────────────────────────────────

MOCK_CRM = {
    'C-1001': {
        'name': 'Priya Sharma',
        'billing_status': 'Active',
        'subscription_tier': 'Enterprise',
        'last_payment_date': '2026-04-01',
        'last_payment_amount': 4999.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-04-01', 'description': 'Enterprise Plan — April',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-03-01', 'description': 'Enterprise Plan — March',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-02-01', 'description': 'Enterprise Plan — February','amount': 4999.00, 'status': 'paid'},
        ],
        'internal_crm_id': 'CRM-88123',
        'sales_rep_code': 'SR-042',
        'geo_region_tag': 'APAC',
        'last_login_ip': '192.168.10.5',
        'feature_flag_cohort': 'beta-v2',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1002': {
        'name': 'Arjun Mehta',
        'billing_status': 'Past Due',
        'subscription_tier': 'Pro',
        'last_payment_date': '2026-03-01',
        'last_payment_amount': 499.00,
        'outstanding_balance': 998.00,
        'recent_transactions': [
            {'date': '2026-03-01', 'description': 'Pro Plan — March',  'amount': 499.00, 'status': 'paid'},
            {'date': '2026-04-01', 'description': 'Pro Plan — April',  'amount': 499.00, 'status': 'missed'},
            {'date': '2026-05-01', 'description': 'Pro Plan — May',    'amount': 499.00, 'status': 'missed'},
        ],
        'internal_crm_id': 'CRM-88456',
        'sales_rep_code': 'SR-017',
        'geo_region_tag': 'APAC',
        'last_login_ip': '10.0.0.44',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1003': {
        'name': 'Kavya Nair',
        'billing_status': 'Active',
        'subscription_tier': 'Starter',
        'last_payment_date': '2026-05-01',
        'last_payment_amount': 99.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-05-01', 'description': 'Starter Plan — May', 'amount': 99.00, 'status': 'paid'},
        ],
        'internal_crm_id': 'CRM-88789',
        'sales_rep_code': 'SR-031',
        'geo_region_tag': 'EMEA',
        'last_login_ip': '172.16.0.9',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
}

# ... entire file continues unchanged below ...
