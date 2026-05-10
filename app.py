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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg-onyx: #0a0a0c;
  --bg-card: rgba(255, 255, 255, 0.03);
  --border-glass: rgba(255, 255, 255, 0.08);
  --text-main: #f3f4f6;
  --text-muted: #8b949e;
  --alert-red: #ff3366;
  --cyber-cyan: #00f0ff;
  --warn-yellow: #ffb800;
  --font-mono: 'JetBrains Mono', monospace;
  --font-primary: 'Space Grotesk', sans-serif;
}

.gradio-container {
  max-width: 1440px !important;
  font-family: var(--font-primary) !important;
  background-color: var(--bg-onyx) !important;
  background-image: 
    linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
  background-size: 24px 24px;
}

#uplan-header {
  border-bottom: 1px solid var(--border-glass);
  padding: 16px 0 20px;
  margin-bottom: 16px;
}
#uplan-header h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: -0.5px;
  color: var(--text-main);
  font-weight: 700;
  text-transform: uppercase;
}
#uplan-header p {
  margin: 6px 0 0;
  color: var(--text-muted);
  font-size: 13px;
  font-family: var(--font-mono);
  letter-spacing: 1px;
}

/* Animations */
@keyframes compileIn {
  0% { opacity: 0; transform: translateY(15px); }
  100% { opacity: 1; transform: translateY(0); }
}

.dossier-card {
  border: 1px solid var(--border-glass);
  border-radius: 4px;
  padding: 16px;
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  color: var(--text-main);
  animation: compileIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
  opacity: 0;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.dossier-card:hover {
  border-color: rgba(255, 255, 255, 0.15);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.card-1 { animation-delay: 0.1s; }
.card-2 { animation-delay: 0.2s; }
.card-3 { animation-delay: 0.3s; }
.card-4 { animation-delay: 0.4s; }
.card-5 { animation-delay: 0.5s; }
.card-6 { animation-delay: 0.6s; }

.dossier-card-title {
  color: var(--text-main);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 1.5px;
  margin-bottom: 12px;
  display: block;
}

/* Layout */
.dashboard-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.hero-section {
  display: grid;
  grid-template-columns: 1fr 2.5fr;
  gap: 16px;
}

.split-view {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 16px;
}

@media (max-width: 900px) {
  .hero-section, .split-view {
    grid-template-columns: 1fr;
  }
}

/* Typography & Elements */
.mono-text {
  font-family: var(--font-mono);
}

.dossier-badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 2px;
  font-size: 10px;
  font-family: var(--font-mono);
  font-weight: 600;
  text-transform: uppercase;
  margin: 2px 4px 2px 0;
  transition: filter 0.2s;
}
.dossier-badge:hover {
  filter: brightness(1.3);
}

.sev-critical { background: rgba(255, 51, 102, 0.1); color: var(--alert-red); border: 1px solid rgba(255, 51, 102, 0.3); }
.sev-warning { background: rgba(255, 184, 0, 0.1); color: var(--warn-yellow); border: 1px solid rgba(255, 184, 0, 0.3); }
.sev-info { background: rgba(255, 255, 255, 0.05); color: var(--text-muted); border: 1px solid var(--border-glass); }
.sev-pass { background: rgba(0, 240, 255, 0.1); color: var(--cyber-cyan); border: 1px solid rgba(0, 240, 255, 0.3); }

.data-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 12px;
}
.data-label {
  color: var(--text-muted);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-family: var(--font-mono);
  display: block;
  margin-bottom: 4px;
}
.data-value {
  color: var(--text-main);
  font-size: 14px;
  font-family: var(--font-mono);
  font-weight: 500;
}

.dossier-row {
  display: flex;
  justify-content: space-between;
  padding: 10px 8px;
  border-bottom: 1px solid var(--border-glass);
  font-family: var(--font-mono);
  font-size: 12px;
  transition: background 0.2s;
  margin: 0 -8px;
}
.dossier-row:hover {
  background: rgba(255, 255, 255, 0.03);
}
.dossier-row:last-child {
  border-bottom: none;
}

.progress-bar-bg {
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 8px;
}
.progress-bar-fill {
  height: 100%;
  transition: width 1s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.text-critical { color: var(--alert-red) !important; }
.text-warning { color: var(--warn-yellow) !important; }
.text-pass { color: var(--cyber-cyan) !important; }
.text-muted { color: var(--text-muted) !important; }

.tool-card {
  border: 1px solid var(--border-glass);
  border-radius: 4px;
  padding: 14px;
  background: var(--bg-card);
  backdrop-filter: blur(8px);
  color: var(--text-main);
  font-family: var(--font-mono);
}
.tool-card b {
  color: var(--text-main);
}
.tool-card hr {
  border: 0;
  border-top: 1px solid var(--border-glass);
  margin: 10px 0;
}

/* ── Chat styling ────────────────────────── */
.gradio-container textarea {
  min-height: 44px !important;
}
.gradio-container .gap {
  gap: 8px;
}
"""


def badge(text: str, severity: str) -> str:
    sev_class = f"sev-{severity}" if severity in ["critical", "warning", "pass"] else "sev-info"
    return f"<span class='dossier-badge {sev_class}'>{text}</span>"


def money(value: Any, currency: str = "") -> str:
    if value is None:
        return "<span class='data-value text-muted'>N/A</span>"
    try:
        return f"<span class='data-value'>{currency} {float(value):,.0f}</span>"
    except (TypeError, ValueError):
        return f"<span class='data-value'>{value}</span>"


def score_bar(score: float) -> str:
    pct = max(0, min(100, int(score * 100)))
    color_class = "text-critical" if pct < 50 else "text-warning" if pct < 75 else "text-pass"
    bg_color = "var(--alert-red)" if pct < 50 else "var(--warn-yellow)" if pct < 75 else "var(--cyber-cyan)"
    return f"""
<div style='margin-top:12px'>
  <div style='display:flex;justify-content:space-between;font-size:11px;margin-bottom:6px;' class='mono-text'>
    <span class='text-muted'>NARRATIVE COHERENCE</span><b class='{color_class}'>{pct}%</b>
  </div>
  <div class='progress-bar-bg'>
    <div class='progress-bar-fill' style='width:{pct}%;background:{bg_color}'></div>
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
        return "<p class='mono-text text-muted' style='padding:16px'>Awaiting telemetry...</p>"

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

    readiness_html = f"""
    <div class='dossier-card card-1'>
      <div style='display:flex;justify-content:space-between;align-items:center'>
        <span class='dossier-card-title' style='margin:0'>Readiness overview</span>{badge(overall, overall_sev)}
      </div>
      {score_bar(score)}
      {"<div class='mono-text text-critical' style='font-size:11px;margin-top:10px;font-weight:600'>[!] HUMAN REVIEW REQUIRED</div>" if synthesis.get("human_review_required") else ""}
    </div>
    """

    financials_html = f"""
    <div class='dossier-card card-2'>
      <span class='dossier-card-title'>Financial summary</span>
      <div class='data-grid'>
        <div><span class='data-label'>Required funds</span>{money(t_req, currency)}</div>
        <div><span class='data-label'>Liquid / deposit evidence</span>{money(account_total or fields.get("balance_closing"), currency)}</div>
        <div><span class='data-label'>Affidavit income</span>{money(fields.get("i_aff") or income_total, currency)}</div>
        <div><span class='data-label'>Tax verified income</span>{money(fields.get("i_tax"), currency)}</div>
      </div>
    </div>
    """
    
    hero_section = f"<div class='hero-section'>{readiness_html}{financials_html}</div>"

    left_col_html = ""
    if findings:
        rows = "".join(
            f"<div class='dossier-row' style='flex-direction:column;align-items:flex-start'>"
            f"<div style='margin-bottom:6px;display:flex;align-items:center;gap:8px'><b style='color:var(--text-main);font-size:13px;font-family:var(--font-primary)'>{finding.get('rule_id','rule').replace('_',' ').title()}</b> {badge(finding.get('severity','info').title(), finding.get('severity','info'))}</div>"
            f"<div style='color:var(--text-muted);line-height:1.5'>{finding.get('message','')}</div>"
            f"</div>"
            for finding in findings
        )
        left_col_html += f"<div class='dossier-card card-3'><span class='dossier-card-title'>Agent findings</span>{rows}</div>"

    if synthesis.get("compound_flags"):
        rows = "".join(f"<div class='dossier-row' style='color:var(--warn-yellow);border-color:rgba(255,184,0,0.1)'>[FLAG] {item}</div>" for item in synthesis["compound_flags"])
        left_col_html += f"<div class='dossier-card card-5' style='border-color:rgba(255,184,0,0.3);background:rgba(255,184,0,0.02)'><span class='dossier-card-title' style='color:var(--warn-yellow)'>Cross-document risks</span>{rows}</div>"

    adversarial = result.get("adversarial_audit", {})
    if adversarial.get("rejection_case") or adversarial.get("rebuttal_case"):
        rows = ""
        if adversarial.get("rejection_case"):
            rows += f"<div class='dossier-row' style='flex-direction:column;align-items:flex-start'><span class='data-label text-critical'>Officer rejection case</span><div style='color:var(--text-muted);margin-top:4px;line-height:1.5'>{adversarial['rejection_case']}</div></div>"
        if adversarial.get("rebuttal_case"):
            rows += f"<div class='dossier-row' style='flex-direction:column;align-items:flex-start'><span class='data-label text-pass'>Applicant rebuttal path</span><div style='color:var(--text-muted);margin-top:4px;line-height:1.5'>{adversarial['rebuttal_case']}</div></div>"
        left_col_html += f"<div class='dossier-card card-6'><span class='dossier-card-title'>Adversarial audit</span>{rows}</div>"

    right_col_html = ""
    
    parties_html = f"""
    <div class='dossier-card card-3'>
      <span class='dossier-card-title'>Parties</span>
      <div class='data-grid'>
        <div><span class='data-label'>Applicant</span><b class='data-value'>{result.get("applicant_name") or fields.get("beneficiary_name") or "not found"}</b></div>
        <div><span class='data-label'>Sponsor</span><b class='data-value'>{result.get("sponsor_name") or "not found"}</b></div>
      </div>
      <div style='margin-top:16px'><span class='data-label'>Relationship</span><b class='data-value'>{result.get("sponsor_relationship") or fields.get("spon_relationship") or "not found"}</b></div>
    </div>
    """
    right_col_html += parties_html

    if docs:
        rows = "".join(
            f"<div class='dossier-row'>"
            f"<span style='color:var(--text-main)'>{doc.get('type','unknown').replace('_',' ').title()}</span>"
            f"<span style='color:var(--text-muted)'>{doc.get('pages',0)}p · {doc.get('quality','unknown')} · {badge(doc.get('status','seen'), 'pass')}</span></div>"
            for doc in docs
        )
        right_col_html += f"<div class='dossier-card card-4'><span class='dossier-card-title'>Documents</span>{rows}</div>"

    if fields.get("name_variants"):
        rows = "".join(
            f"<div class='dossier-row'><span class='data-label' style='margin:0'>{key.replace('_',' ').title()}</span><span class='data-value' style='font-size:12px'>{value}</span></div>"
            for key, value in fields["name_variants"].items()
        )
        right_col_html += f"<div class='dossier-card card-4'><span class='dossier-card-title'>Name consistency</span>{rows}</div>"

    if result.get("next_steps"):
        rows = "".join(
            f"<div class='dossier-row' style='align-items:flex-start;gap:12px'>{badge(step.get('priority','info').upper(), step.get('priority','info'))}<span style='color:var(--text-muted);flex:1;line-height:1.4'>{step.get('action','')}</span></div>"
            for step in result["next_steps"]
        )
        right_col_html += f"<div class='dossier-card card-5'><span class='dossier-card-title'>Next steps</span>{rows}</div>"

    cert = result.get("deletion_cert") or raw_result.get("full_result", {}).get("deletion_cert")
    if cert:
        right_col_html += f"<div class='dossier-card card-6' style='background:transparent;border:1px dashed var(--border-glass)'><span class='dossier-card-title' style='color:var(--text-muted)'>Deletion certificate</span><pre class='mono-text' style='white-space:pre-wrap;font-size:10px;color:var(--text-muted);margin:0;line-height:1.4'>{cert}</pre></div>"

    split_view = f"<div class='split-view'><div class='left-col' style='display:flex;flex-direction:column;gap:16px'>{left_col_html}</div><div class='right-col' style='display:flex;flex-direction:column;gap:16px'>{right_col_html}</div></div>"

    return f"<div class='dashboard-grid'>{hero_section}{split_view}</div>"


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
  <h1>Uplan / Command</h1>
  <p>IMMIGRATION DOCUMENT INTELLIGENCE // ADVERSARIAL AUDIT DOSSIER</p>
</div>
"""


CHECKLIST_HTML = """
<div class='tool-card'>
  <b style='font-size:11px;letter-spacing:1px;text-transform:uppercase'>[ LOG ] Packet checklist</b><br><br>
  <span class='text-critical'>REQ:</span> affidavit, bank statement, tax return<br>
  <span class='text-warning'>REC:</span> bank balance certificate<br>
  <span class='text-pass'>OPT:</span> passport / identity document
  <hr>
  <b style='font-size:11px;letter-spacing:1px;text-transform:uppercase'>[ SYS ] Agent tools</b><br><br>
  <span class='text-muted'>AUDITOR:</span> document readiness and quality<br>
  <span class='text-muted'>STRATEGIST:</span> roadmap and missing-evidence<br>
  <span class='text-muted'>SYNTHESIS:</span> cross-document risk narrative
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
        theme=gr.themes.Base(),
        css=CSS,
    )

