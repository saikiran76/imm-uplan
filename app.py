from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr


# USE_LIVE_BACKEND = os.getenv("USE_LIVE_BACKEND", "false").lower() == "true"
USE_LIVE_BACKEND = True
AMD_ENDPOINT = os.getenv("AMD_ENDPOINT", "http://165.245.142.235:8000")
UPLAN_API_KEY = os.getenv("UPLAN_API_KEY", "")
SAMPLE_PATH = Path(__file__).parent / "sample_outputs" / "demo_result.json"
DEMO_RESULT = json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else {}


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;450;500;600;700&display=swap');

:root {
  --uplan-bg: #0c0d10;
  --uplan-surface: rgba(255,255,255,0.035);
  --uplan-surface-hover: rgba(255,255,255,0.06);
  --uplan-border: rgba(255,255,255,0.07);
  --uplan-border-accent: rgba(96,165,250,0.2);
  --uplan-text: rgb(219,216,211);
  --uplan-text-muted: rgba(219,216,211,0.55);
  --uplan-text-dim: rgba(219,216,211,0.35);
  --uplan-accent-1: #60a5fa;
  --uplan-accent-2: #a78bfa;
  --uplan-accent-3: #818cf8;
  --uplan-green: #34d399;
  --uplan-amber: #fbbf24;
  --uplan-red: #f87171;
  --uplan-radius: 14px;
  --uplan-radius-sm: 10px;
  --font-main: "Google Sans Flex", "Google Sans", "Inter", sans-serif;
}

/* ── Global Reset ────────────────────────── */
.gradio-container {
  max-width: 1440px !important;
  font-family: var(--font-main) !important;
  font-weight: 450 !important;
  color: var(--uplan-text) !important;
  background: var(--uplan-bg) !important;
  font-size: 14px !important;
  line-height: 1.6 !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.gradio-container .dark {
  background: var(--uplan-bg) !important;
}

/* ── Header ──────────────────────────────── */
#uplan-header {
  border-bottom: 1px solid var(--uplan-border);
  padding: 28px 0 24px;
  margin-bottom: 20px;
  position: relative;
}
#uplan-header::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 120px;
  height: 2px;
  background: linear-gradient(90deg, var(--uplan-accent-1), var(--uplan-accent-2));
  border-radius: 2px;
}
#uplan-header h1 {
  margin: 0;
  font-size: 54px;
  line-height: 56px;
  letter-spacing: -1.5px;
  background: linear-gradient(135deg, var(--uplan-accent-1), var(--uplan-accent-2), var(--uplan-accent-3));
  background-size: 200% 200%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
  animation: uplan-shimmer 6s ease-in-out infinite;
}
@keyframes uplan-shimmer {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}
#uplan-header p {
  margin: 10px 0 0;
  color: var(--uplan-text-muted);
  font-size: 15px;
  font-weight: 400;
  max-width: 580px;
  letter-spacing: 0.01em;
}

/* ── Tool cards (sidebar) ────────────────── */
.tool-card {
  border: 1px solid var(--uplan-border);
  border-radius: var(--uplan-radius);
  padding: 18px 20px;
  background: var(--uplan-surface);
  backdrop-filter: blur(20px) saturate(1.2);
  -webkit-backdrop-filter: blur(20px) saturate(1.2);
  color: var(--uplan-text);
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.tool-card:hover {
  border-color: var(--uplan-border-accent);
  box-shadow: 0 0 24px rgba(96,165,250,0.06);
}
.tool-card b {
  color: rgba(255,255,255,0.95);
  font-weight: 600;
}
.tool-card hr {
  border: 0;
  border-top: 1px solid var(--uplan-border);
  margin: 14px 0;
}

/* ── Chat styling ────────────────────────── */
.gradio-container textarea {
  min-height: 48px !important;
  font-family: var(--font-main) !important;
  font-weight: 450 !important;
  border-radius: var(--uplan-radius-sm) !important;
  border: 1px solid var(--uplan-border) !important;
  background: var(--uplan-surface) !important;
  transition: border-color 0.25s ease !important;
}
.gradio-container textarea:focus {
  border-color: var(--uplan-accent-1) !important;
  box-shadow: 0 0 0 3px rgba(96,165,250,0.1) !important;
}
.gradio-container .gap {
  gap: 10px;
}

/* ── Buttons ─────────────────────────────── */
.gradio-container button.primary {
  background: linear-gradient(135deg, var(--uplan-accent-1), var(--uplan-accent-2)) !important;
  border: none !important;
  border-radius: var(--uplan-radius-sm) !important;
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  padding: 10px 24px !important;
  transition: transform 0.2s ease, box-shadow 0.25s ease !important;
  box-shadow: 0 4px 16px rgba(96,165,250,0.15) !important;
}
.gradio-container button.primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 28px rgba(96,165,250,0.25) !important;
}
.gradio-container button.secondary {
  border-radius: var(--uplan-radius-sm) !important;
  font-family: var(--font-main) !important;
  font-weight: 500 !important;
  border: 1px solid var(--uplan-border) !important;
  background: var(--uplan-surface) !important;
  transition: background 0.2s ease, border-color 0.2s ease !important;
}
.gradio-container button.secondary:hover {
  background: var(--uplan-surface-hover) !important;
  border-color: var(--uplan-accent-1) !important;
}

/* ── Chatbot bubble override ─────────────── */
.gradio-container .chatbot .message {
  font-family: var(--font-main) !important;
  font-weight: 450 !important;
  border-radius: var(--uplan-radius) !important;
}

/* ── Labels & markdown ───────────────────── */
.gradio-container label, .gradio-container .label-wrap span {
  font-family: var(--font-main) !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em;
}
.gradio-container .prose h3, .gradio-container .markdown h3 {
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  font-size: 18px !important;
  letter-spacing: -0.3px;
  color: var(--uplan-text) !important;
}

/* ── File upload ─────────────────────────── */
.gradio-container .upload-area {
  border: 1px dashed var(--uplan-border) !important;
  border-radius: var(--uplan-radius) !important;
  background: var(--uplan-surface) !important;
  transition: border-color 0.25s ease !important;
}
.gradio-container .upload-area:hover {
  border-color: var(--uplan-accent-1) !important;
}

/* ── Scrollbar ───────────────────────────── */
.gradio-container ::-webkit-scrollbar { width: 6px; }
.gradio-container ::-webkit-scrollbar-track { background: transparent; }
.gradio-container ::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
}
.gradio-container ::-webkit-scrollbar-thumb:hover {
  background: rgba(255,255,255,0.18);
}

/* ── Animations ──────────────────────────── */
@keyframes uplan-fade-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.uplan-dash-card {
  animation: uplan-fade-in 0.4s ease-out both;
}
"""


SEV_COLOR = {
    "critical": ("rgba(239,68,68,0.15)", "#f87171"),
    "warning": ("rgba(245,158,11,0.15)", "#fbbf24"),
    "info": ("rgba(96,165,250,0.15)", "#60a5fa"),
    "pass": ("rgba(52,211,153,0.15)", "#34d399"),
}


def badge(text: str, severity: str) -> str:
    bg, color = SEV_COLOR.get(severity, ("rgba(255,255,255,0.06)", "rgba(255,255,255,0.6)"))
    return (
        f"<span style='display:inline-flex;align-items:center;padding:4px 12px;border-radius:999px;"
        f"font-size:11px;font-weight:600;background:{bg};color:{color};"
        f"border:1px solid {color}22;letter-spacing:0.03em;"
        f"font-family:\"Google Sans Flex\",\"Inter\",sans-serif;"
        f"backdrop-filter:blur(8px);"
        f"margin:2px 4px 2px 0;transition:all 0.2s ease'>{text}</span>"
    )


def money(value: Any, currency: str = "") -> str:
    if value is None:
        return "<span style='color:rgba(219,216,211,0.3);font-style:italic;font-size:13px'>not found</span>"
    try:
        return (
            f"<b style='color:rgb(219,216,211);font-weight:600;font-size:16px;"
            f"font-family:\"Google Sans Flex\",\"Inter\",sans-serif'>"
            f"{currency} {float(value):,.0f}</b>"
        )
    except (TypeError, ValueError):
        return f"<b style='color:rgb(219,216,211);font-weight:600'>{value}</b>"


def score_bar(score: float) -> str:
    pct = max(0, min(100, int(score * 100)))
    color = "#f87171" if pct < 50 else "#fbbf24" if pct < 75 else "#34d399"
    glow = f"0 0 12px {color}44"
    return f"""
<div style='margin-top:14px'>
  <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px;color:rgba(219,216,211,0.6);font-family:"Google Sans Flex","Inter",sans-serif'>
    <span style='font-weight:500'>Narrative coherence</span>
    <b style='color:{color};font-size:18px;font-weight:700'>{pct}<span style='font-size:12px;font-weight:500;opacity:0.7'>%</span></b>
  </div>
  <div style='height:6px;background:rgba(255,255,255,0.06);border-radius:999px;overflow:hidden;position:relative'>
    <div style='height:6px;width:{pct}%;background:linear-gradient(90deg,{color}99,{color});border-radius:999px;box-shadow:{glow};transition:width 0.8s cubic-bezier(0.4,0,0.2,1)'></div>
  </div>
</div>
"""


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    if not result:
        return {}

    if "agent_output" in result:
        output = result.get("agent_output", {})
        fields = result.get("reliable_fields", {})
        findings = result.get("findings", output.get("findings", []))
        full = result.get("full_result", {})
        summary = result.get("summary", {})
        narrative = result.get("narrative_synthesis") or {
            "narrative_score": output.get("narrative_score"),
            "human_review_required": output.get("human_review_required"),
            "synthesis_trace": output.get("synthesis_trace"),
        }
        adversarial = result.get("adversarial_audit") or {
            "rejection_case": output.get("rejection_case"),
            "rebuttal_case": output.get("rebuttal_case"),
        }
        return {
            "applicant_name": fields.get("beneficiary_name") or "Applicant",
            "sponsor_name": fields.get("name_variants", {}).get("affidavit")
                or fields.get("name_variants", {}).get("bank_balance_certificate", "Sponsor"),
            "sponsor_relationship": fields.get("spon_relationship"),
            "currency_code": fields.get("currency_code"),
            "t_req": output.get("t_req", 800000),
            "documents_parsed": infer_documents(full, summary),
            "reliable_fields": fields,
            "agent_findings": findings,
            "narrative_synthesis": narrative,
            "adversarial_audit": adversarial,
            "next_steps": next_steps(findings),
            "deletion_cert": full.get("deletion_cert", ""),
        }

    if "full_result" in result and "agent_findings" not in result:
        fields = result.get("reliable_fields", {})
        full = result.get("full_result", {})
        return {
            "applicant_name": fields.get("beneficiary_name") or "Applicant",
            "sponsor_name": fields.get("name_variants", {}).get("affidavit", "Sponsor"),
            "sponsor_relationship": fields.get("spon_relationship"),
            "currency_code": fields.get("currency_code"),
            "t_req": 800000,
            "documents_parsed": infer_documents(full, result.get("summary", {})),
            "reliable_fields": fields,
            "agent_findings": [],
            "narrative_synthesis": synthesize([]),
            "next_steps": next_steps([]),
            "deletion_cert": full.get("deletion_cert", ""),
        }

    return result


def infer_documents(full: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    pages = full.get("pages", [])
    if not pages:
        total = summary.get("total_pages", 0)
        return [{"type": "packet", "pages": total, "quality": summary.get("source_quality", "unknown"), "status": "extracted"}]
    docs: dict[str, dict[str, Any]] = {}
    for page in pages:
        ptype = page.get("page_type", "unknown")
        row = docs.setdefault(ptype, {"type": ptype, "pages": 0, "quality": page.get("source_quality", "unknown"), "status": "extracted"})
        row["pages"] += 1
    return list(docs.values())


def synthesize(findings: list[dict[str, Any]]) -> dict[str, Any]:
    critical = sum(1 for item in findings if item.get("severity") == "critical")
    warning = sum(1 for item in findings if item.get("severity") == "warning")
    score = max(0.2, 0.88 - critical * 0.22 - warning * 0.08)
    return {
        "narrative_score": score,
        "human_review_required": critical > 0 or any(item.get("requires_human_review") for item in findings),
        "compound_flags": [item["message"] for item in findings[:2]],
        "synthesis_trace": "Agents reviewed liquidity, income coherence, and document corroboration. Missing cross-document evidence reduces readiness confidence.",
    }


def next_steps(findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not findings:
        return [
            {"priority": "critical", "action": "Upload bank statements and tax return to enable cross-document checks."},
            {"priority": "warning", "action": "Add bank balance certificate to corroborate closing balances."},
        ]
    steps = []
    for finding in findings[:4]:
        sev = finding.get("severity", "info")
        steps.append({"priority": sev, "action": finding.get("message", "Review this finding.")})
    return steps


def run_backend(files: list[Any] | None) -> dict[str, Any]:
    if not files:
        return backend_unavailable_result("No files were uploaded.")

    if not USE_LIVE_BACKEND:
        return backend_unavailable_result(
            "Live backend is disabled. Set USE_LIVE_BACKEND=true and AMD_ENDPOINT to the Flask /extract URL."
        )

    import requests

    upload_files = []
    handles = []
    headers = {}
    if UPLAN_API_KEY:
        headers["X-API-Key"] = UPLAN_API_KEY
    try:
        for file_obj in files:
            path = getattr(file_obj, "name", file_obj)
            handle = open(path, "rb")
            handles.append(handle)
            upload_files.append(("files", (Path(path).name, handle, "application/pdf")))
        url = f"{AMD_ENDPOINT.rstrip('/')}/extract"
        response = requests.post(url, files=upload_files, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return backend_unavailable_result(
            f"Live backend did not respond at {AMD_ENDPOINT}: {exc}"
        )
    finally:
        for handle in handles:
            handle.close()


def backend_unavailable_result(message: str) -> dict[str, Any]:
    return {
        "applicant_name": "Live backend unavailable",
        "sponsor_name": "Not analysed",
        "sponsor_relationship": None,
        "currency_code": "",
        "t_req": 800000,
        "documents_parsed": [
            {"type": "upload_received", "pages": 0, "quality": "not_processed", "status": "backend_offline"}
        ],
        "reliable_fields": {
            "currency_code": None,
            "name_variants": {},
            "financial_accounts": [],
            "income_sources": [],
            "properties": [],
            "movable_assets": [],
        },
        "agent_findings": [
            {
                "agent_id": "system",
                "rule_id": "LIVE_BACKEND_UNAVAILABLE",
                "severity": "critical",
                "message": message,
                "requires_human_review": True,
            }
        ],
        "narrative_synthesis": {
            "narrative_score": 0.0,
            "human_review_required": True,
            "compound_flags": ["Uploaded PDFs were not analysed because the live backend is offline or disabled."],
            "synthesis_trace": "No extraction or agent graph run occurred for this upload.",
        },
        "adversarial_audit": {
            "rejection_case": "No live backend result exists for this upload.",
            "rebuttal_case": "Start the AMD Flask backend and rerun the packet.",
        },
        "next_steps": [
            {"priority": "critical", "action": "Start the AMD Flask backend and set USE_LIVE_BACKEND=true in the HF Space."}
        ],
        "backend_status": "offline",
    }


def build_dashboard_html(raw_result: dict[str, Any]) -> str:
    result = normalize_result(raw_result)
    if not result:
        return "<div style='text-align:center;padding:48px 16px;color:rgba(219,216,211,0.35);font-family:\"Google Sans Flex\",\"Inter\",sans-serif'><div style='font-size:40px;margin-bottom:12px;opacity:0.4'>📄</div><p style='font-size:15px;font-weight:450'>Upload documents or run demo mode to begin.</p></div>"

    fields = result.get("reliable_fields", {})
    synthesis = result.get("narrative_synthesis", {})
    findings = result.get("agent_findings", [])
    docs = result.get("documents_parsed", [])
    currency = result.get("currency_code") or fields.get("currency_code") or ""
    t_req = result.get("t_req", 800000)
    score = synthesis.get("narrative_score", 0.0)
    overall = "Critical" if score < 0.5 else "Advisory" if score < 0.75 else "Pass"
    overall_sev = "critical" if score < 0.5 else "warning" if score < 0.75 else "pass"
    account_total = sum(float(item.get("amount") or 0) for item in fields.get("financial_accounts", []))
    income_total = sum(float(item.get("annual_amount") or 0) for item in fields.get("income_sources", []))

    font = '"Google Sans Flex","Inter",sans-serif'
    label_style = f"color:rgba(219,216,211,0.4);font-size:10px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;font-family:{font}"
    card_bg = "rgba(255,255,255,0.03)"
    card_border = "rgba(255,255,255,0.07)"

    human_review_html = ""
    if synthesis.get("human_review_required"):
        human_review_html = f"<div style='display:flex;align-items:center;gap:8px;font-size:12px;color:#f87171;margin-top:10px;padding:8px 12px;background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.12);border-radius:8px;font-family:{font}'><span style='font-size:16px'>⚠</span> <span style='font-weight:500'>Human review required</span></div>"

    text_primary = "rgb(219,216,211)"
    accent_bg = "linear-gradient(135deg,rgba(96,165,250,0.06),rgba(167,139,250,0.06))"
    sub_card = "background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:12px 14px"

    html = f"""
<div style='font-family:{font};font-weight:450;font-size:14px;line-height:1.6;color:{text_primary}'>

  <div class='uplan-dash-card' style='background:{accent_bg};border:1px solid {card_border};border-radius:14px;padding:20px;margin-bottom:14px;backdrop-filter:blur(20px)'>
    <div style='display:flex;justify-content:space-between;align-items:center'>
      <div style='display:flex;align-items:center;gap:10px'>
        <span style='font-size:20px'>📊</span>
        <b style='color:{text_primary};font-size:15px;font-weight:600'>Readiness overview</b>
      </div>
      {badge(overall, overall_sev)}
    </div>
    {score_bar(score)}
    {human_review_html}
  </div>

  <div class='uplan-dash-card' style='background:{card_bg};border:1px solid {card_border};border-radius:14px;padding:20px;margin-bottom:14px;backdrop-filter:blur(16px)'>
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>
      <span style='font-size:18px'>👤</span>
      <b style='color:{text_primary};font-size:15px;font-weight:600'>Parties</b>
    </div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:14px'>
      <div style='{sub_card}'>
        <span style='{label_style}'>Applicant</span>
        <div style='margin-top:6px;font-weight:500;color:{text_primary}'>{result.get("applicant_name") or fields.get("beneficiary_name") or "not found"}</div>
      </div>
      <div style='{sub_card}'>
        <span style='{label_style}'>Sponsor</span>
        <div style='margin-top:6px;font-weight:500;color:{text_primary}'>{result.get("sponsor_name") or "not found"}</div>
      </div>
    </div>
    <div style='margin-top:10px;color:rgba(219,216,211,0.5);font-size:13px'>Relationship: <b style='color:{text_primary};font-weight:500'>{result.get("sponsor_relationship") or fields.get("spon_relationship") or "not found"}</b></div>
  </div>

  <div class='uplan-dash-card' style='background:{card_bg};border:1px solid {card_border};border-radius:14px;padding:20px;margin-bottom:14px;backdrop-filter:blur(16px)'>
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>
      <span style='font-size:18px'>💰</span>
      <b style='color:{text_primary};font-size:15px;font-weight:600'>Financial summary</b>
    </div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>
      <div style='{sub_card}'><span style='{label_style}'>Required funds</span><div style='margin-top:6px'>{money(t_req, currency)}</div></div>
      <div style='{sub_card}'><span style='{label_style}'>Liquid / deposit evidence</span><div style='margin-top:6px'>{money(account_total or fields.get("balance_closing"), currency)}</div></div>
      <div style='{sub_card}'><span style='{label_style}'>Affidavit income</span><div style='margin-top:6px'>{money(fields.get("i_aff") or income_total, currency)}</div></div>
      <div style='{sub_card}'><span style='{label_style}'>Tax verified income</span><div style='margin-top:6px'>{money(fields.get("i_tax"), currency)}</div></div>
    </div>
  </div>
"""

    if docs:
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>"
            f"<span style='color:rgb(219,216,211);font-weight:500'>{doc.get('type','unknown').replace('_',' ').title()}</span>"
            f"<span style='display:flex;align-items:center;gap:8px;color:rgba(219,216,211,0.4);font-size:12px'>{doc.get('pages',0)}p · {doc.get('quality','unknown')} {badge(doc.get('status','seen'), 'pass')}</span></div>"
            for doc in docs
        )
        html += card("📑 Documents", rows)

    if fields.get("name_variants"):
        rows = "".join(
            f"<div style='font-size:13px;color:rgba(219,216,211,0.6);padding:4px 0'><span style='color:rgba(219,216,211,0.35);font-weight:500'>{key.replace('_',' ').title()}</span>: <span style='color:rgb(219,216,211);font-weight:500'>{value}</span></div>"
            for key, value in fields["name_variants"].items()
        )
        html += card("🔗 Name consistency", rows)

    if findings:
        rows = "".join(
            f"<div style='padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>"
            f"<div style='display:flex;align-items:center;gap:8px'><b style='color:rgb(219,216,211);font-weight:600'>{finding.get('rule_id','rule').replace('_',' ').title()}</b> {badge(finding.get('severity','info').title(), finding.get('severity','info'))}</div>"
            f"<div style='font-size:12px;color:rgba(219,216,211,0.45);margin-top:4px;line-height:1.5'>{finding.get('message','')}</div>"
            f"</div>"
            for finding in findings
        )
        html += card("🔍 Agent findings", rows)

    if synthesis.get("compound_flags"):
        rows = "".join(f"<div style='font-size:13px;padding:5px 0;color:rgba(219,216,211,0.7);line-height:1.5'>{item}</div>" for item in synthesis["compound_flags"])
        html += card("⚡ Cross-document risks", rows, bg="rgba(245,158,11,0.05)", border="rgba(245,158,11,0.15)")

    adversarial = result.get("adversarial_audit", {})
    if adversarial.get("rejection_case") or adversarial.get("rebuttal_case"):
        rows = ""
        if adversarial.get("rejection_case"):
            rows += f"<div style='font-size:13px;padding:6px 0;color:rgba(219,216,211,0.7);line-height:1.5'><b style='color:rgb(219,216,211);font-weight:600'>Officer rejection case:</b> {adversarial['rejection_case']}</div>"
        if adversarial.get("rebuttal_case"):
            rows += f"<div style='font-size:13px;padding:6px 0;color:rgba(219,216,211,0.7);line-height:1.5'><b style='color:rgb(219,216,211);font-weight:600'>Applicant rebuttal path:</b> {adversarial['rebuttal_case']}</div>"
        html += card("⚖️ Adversarial audit", rows)

    if result.get("next_steps"):
        rows = "".join(
            f"<div style='display:flex;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);align-items:flex-start'>{badge(step.get('priority','info').upper(), step.get('priority','info'))}<span style='color:rgba(219,216,211,0.7);font-size:13px;line-height:1.5'>{step.get('action','')}</span></div>"
            for step in result["next_steps"]
        )
        html += card("🚀 Next steps", rows)

    cert = result.get("deletion_cert") or raw_result.get("full_result", {}).get("deletion_cert")
    if cert:
        html += f"<div class='uplan-dash-card' style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:16px;margin-bottom:6px'><div style='font-size:10px;color:rgba(219,216,211,0.35);text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>🔐 Deletion certificate</div><pre style='white-space:pre-wrap;font-size:11px;color:rgba(219,216,211,0.45);margin-top:8px;font-family:monospace;line-height:1.5'>{cert}</pre></div>"

    html += "</div>"
    return html


def card(title: str, body: str, bg: str = "rgba(255,255,255,0.03)", border: str = "rgba(255,255,255,0.07)") -> str:
    return (
        f"<div class='uplan-dash-card' style='background:{bg};border:1px solid {border};"
        f"border-radius:14px;padding:20px;margin-bottom:14px;"
        f"backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);"
        f"font-family:\"Google Sans Flex\",\"Inter\",sans-serif'>"
        f"<b style='color:rgb(219,216,211);font-size:15px;font-weight:600'>{title}</b>"
        f"<div style='margin-top:12px'>{body}</div></div>"
    )


def chat_response(message: str, history: list, result_state: dict) -> tuple[list, str]:
    result = normalize_result(result_state)
    if not message:
        return history, ""

    findings = result.get("agent_findings", [])
    synthesis = result.get("narrative_synthesis", {})
    adversarial = result.get("adversarial_audit", {})
    if findings:
        first = findings[0]["message"]
        reply = (
            f"The main issue is: {first} "
            f"The current readiness score is {int(float(synthesis.get('narrative_score') or 0) * 100)}%. "
            f"Officer-style concern: {adversarial.get('rejection_case', 'review missing corroborating evidence.')}"
        )
    else:
        reply = (
            "No agent findings are available yet. Run the demo packet or connect the live AMD backend "
            "so the uploaded documents can be extracted and evaluated."
        )

    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}], ""


def handle_upload(files, demo_mode: bool):
    result = DEMO_RESULT if demo_mode else run_backend(files)
    normalized = normalize_result(result)
    findings = normalized.get("agent_findings", [])
    critical = sum(1 for item in findings if item.get("severity") == "critical")
    warning = sum(1 for item in findings if item.get("severity") == "warning")
    intro = (
        f"I analysed the packet and found {critical} critical issue(s) and "
        f"{warning} advisory item(s). The dashboard has been updated. "
        "Ask me about any finding or missing document."
    )
    history = [{"role": "user", "content": "Analysis complete. What are my main issues?"}, {"role": "assistant", "content": intro}]
    return build_dashboard_html(normalized), history, normalized


HEADER_HTML = """
<div id='uplan-header'>
  <h1>⚡ Uplan</h1>
  <p>Immigration document intelligence — reads financial packets, audits evidence quality, flags inconsistencies, and builds a readiness roadmap.<br>
  <span style='font-size:11px;color:rgba(219,216,211,0.3);font-weight:400;letter-spacing:0.04em'>POWERED BY AMD MI300X · QWEN 3.6-27B VLM</span></p>
</div>
"""


CHECKLIST_HTML = """
<div class='tool-card' style='font-size:13px;line-height:2.0'>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>
    <span style='font-size:16px'>📋</span>
    <b style='font-weight:600;font-size:14px'>Packet checklist</b>
  </div>
  <div style='display:flex;align-items:center;gap:6px'><span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:#f87171'></span> <span style='color:#f87171;font-weight:500'>Required:</span> affidavit, bank statement, tax return</div>
  <div style='display:flex;align-items:center;gap:6px'><span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:#fbbf24'></span> <span style='color:#fbbf24;font-weight:500'>Recommended:</span> bank balance certificate</div>
  <div style='display:flex;align-items:center;gap:6px'><span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:#60a5fa'></span> <span style='color:#60a5fa;font-weight:500'>Optional:</span> passport / identity document</div>
  <hr>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>
    <span style='font-size:16px'>🤖</span>
    <b style='font-weight:600;font-size:14px'>Agent tools</b>
  </div>
  <div style='color:rgba(219,216,211,0.65);line-height:1.8'>
    <span style='color:rgba(219,216,211,0.85);font-weight:500'>Auditor</span> · document readiness and quality checks<br>
    <span style='color:rgba(219,216,211,0.85);font-weight:500'>Strategist</span> · roadmap and missing-evidence checklist<br>
    <span style='color:rgba(219,216,211,0.85);font-weight:500'>Synthesis</span> · cross-document risk narrative
  </div>
</div>
"""


UPLAN_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Google Sans Flex"),
        gr.themes.GoogleFont("Inter"),
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("JetBrains Mono"),
        gr.themes.GoogleFont("Fira Code"),
        "monospace",
    ],
)


def build_ui():
    with gr.Blocks(
        title="Uplan - Immigration Document Intelligence",
    ) as app:
        result_state = gr.State({})
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### ⚙️ Agent tools")
                demo_toggle = gr.Checkbox(label="Demo mode", value=True, info="Uses a clearly marked sample analysis. Turn off for live AMD backend.")
                upload_box = gr.File(label="Upload PDF packet", file_types=[".pdf"], file_count="multiple", visible=False)
                demo_toggle.change(lambda enabled: gr.update(visible=not enabled), demo_toggle, upload_box)
                analyse_btn = gr.Button("⚡ Analyse packet", variant="primary")
                gr.HTML(CHECKLIST_HTML)
                gr.Markdown("<span style='font-size:12px;color:rgba(219,216,211,0.35)'>Raw documents are processed outside the UI in live mode. The extraction layer emits a deletion certificate after purge.</span>")

            with gr.Column(scale=2, min_width=360):
                gr.Markdown("### 💬 Consultant")
                chatbot = gr.Chatbot(
                    label="Consultant",
                    height=580,
                    show_label=False,
                    value=[{"role": "assistant", "content": "Welcome to **Uplan**. Load the sample demo or upload a packet when the AMD backend is online. The dashboard will reflect whether the result is live or backend-offline.\n\n*Ask me about risks, missing evidence, or strategy.*"}],
                )
                with gr.Row():
                    chat_input = gr.Textbox(placeholder="Ask about risks, missing evidence, or next steps...", show_label=False, container=False, scale=5)
                    send_btn = gr.Button("Send", variant="secondary", scale=1)

            with gr.Column(scale=2, min_width=400):
                gr.Markdown("### 📊 Readiness dashboard")
                dashboard = gr.HTML("<div style='text-align:center;padding:48px 16px;color:rgba(219,216,211,0.35);font-family:\"Google Sans Flex\",\"Inter\",sans-serif'><div style='font-size:40px;margin-bottom:12px;opacity:0.4'>📄</div><p style='font-size:15px;font-weight:450'>Load demo or upload documents to begin.</p></div>")

        analyse_btn.click(handle_upload, [upload_box, demo_toggle], [dashboard, chatbot, result_state])
        send_btn.click(chat_response, [chat_input, chatbot, result_state], [chatbot, chat_input])
        chat_input.submit(chat_response, [chat_input, chatbot, result_state], [chatbot, chat_input])

    return app


demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=UPLAN_THEME,
        css=CSS,
    )

