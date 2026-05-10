from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr


USE_LIVE_BACKEND = os.getenv("USE_LIVE_BACKEND", "false").lower() == "true"
AMD_ENDPOINT = os.getenv("AMD_ENDPOINT", "http://localhost:8000/extract")
UPLAN_API_KEY = os.getenv("UPLAN_API_KEY", "")
SAMPLE_PATH = Path(__file__).parent / "sample_outputs" / "demo_result.json"
DEMO_RESULT = json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else {}


CSS = """
.gradio-container {
  max-width: 1440px !important;
  font-family: Georgia, 'Times New Roman', serif !important;
}
#uplan-header {
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding: 16px 0 20px;
  margin-bottom: 16px;
}
#uplan-header h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #60a5fa, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
}
#uplan-header p {
  margin: 6px 0 0;
  color: rgba(255,255,255,0.55);
  font-size: 13px;
}
.tool-card {
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px;
  padding: 14px;
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(8px);
  color: rgba(255,255,255,0.85);
}
.tool-card b {
  color: rgba(255,255,255,0.95);
}
.tool-card hr {
  border: 0;
  border-top: 1px solid rgba(255,255,255,0.08);
  margin: 10px 0;
}
/* Chat input fix */
.gradio-container textarea {
  min-height: 44px !important;
}
/* Better spacing for chat row */
.gradio-container .gap {
  gap: 8px;
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
        f"<span style='display:inline-block;padding:3px 10px;border-radius:999px;"
        f"font-size:11px;font-weight:600;background:{bg};color:{color};"
        f"border:1px solid {color}22;"
        f"margin:2px 3px 2px 0'>{text}</span>"
    )


def money(value: Any, currency: str = "") -> str:
    if value is None:
        return "<span style='color:rgba(255,255,255,0.3)'>not found</span>"
    try:
        return f"<b style='color:#e2e8f0'>{currency} {float(value):,.0f}</b>"
    except (TypeError, ValueError):
        return f"<b style='color:#e2e8f0'>{value}</b>"


def score_bar(score: float) -> str:
    pct = max(0, min(100, int(score * 100)))
    color = "#f87171" if pct < 50 else "#fbbf24" if pct < 75 else "#34d399"
    return f"""
<div style='margin-top:10px'>
  <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;color:rgba(255,255,255,0.7)'>
    <span>Narrative coherence</span><b style='color:{color}'>{pct}%</b>
  </div>
  <div style='height:8px;background:rgba(255,255,255,0.08);border-radius:999px;overflow:hidden'>
    <div style='height:8px;width:{pct}%;background:linear-gradient(90deg,{color}cc,{color});border-radius:999px;transition:width 0.5s ease'></div>
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
        response = requests.post(AMD_ENDPOINT, files=upload_files, headers=headers, timeout=300)
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
        return "<p style='color:rgba(255,255,255,0.4);padding:16px'>Upload documents or run demo mode to begin.</p>"

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

    label_style = "color:rgba(255,255,255,0.45);font-size:11px;text-transform:uppercase;letter-spacing:0.5px"
    card_bg = "rgba(255,255,255,0.04)"
    card_border = "rgba(255,255,255,0.08)"

    html = f"""
<div style='font-family:Georgia,"Times New Roman",serif;font-size:13px;line-height:1.6;color:rgba(255,255,255,0.85)'>
  <div style='background:linear-gradient(135deg,rgba(96,165,250,0.08),rgba(167,139,250,0.08));border:1px solid {card_border};border-radius:10px;padding:16px;margin-bottom:12px'>
    <div style='display:flex;justify-content:space-between;align-items:center'>
      <b style='color:#e2e8f0'>Readiness overview</b>{badge(overall, overall_sev)}
    </div>
    {score_bar(score)}
    {"<div style='font-size:12px;color:#f87171;margin-top:8px'>⚠ Human review required</div>" if synthesis.get("human_review_required") else ""}
  </div>

  <div style='background:{card_bg};border:1px solid {card_border};border-radius:10px;padding:16px;margin-bottom:12px'>
    <b style='color:#e2e8f0'>Parties</b>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px'>
      <div><span style='{label_style}'>Applicant</span><br><b style='color:#e2e8f0'>{result.get("applicant_name") or fields.get("beneficiary_name") or "not found"}</b></div>
      <div><span style='{label_style}'>Sponsor</span><br><b style='color:#e2e8f0'>{result.get("sponsor_name") or "not found"}</b></div>
    </div>
    <div style='margin-top:8px;color:rgba(255,255,255,0.6)'>Relationship: <b style='color:#e2e8f0'>{result.get("sponsor_relationship") or fields.get("spon_relationship") or "not found"}</b></div>
  </div>

  <div style='background:{card_bg};border:1px solid {card_border};border-radius:10px;padding:16px;margin-bottom:12px'>
    <b style='color:#e2e8f0'>Financial summary</b>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px'>
      <div><span style='{label_style}'>Required funds</span><br>{money(t_req, currency)}</div>
      <div><span style='{label_style}'>Liquid / deposit evidence</span><br>{money(account_total or fields.get("balance_closing"), currency)}</div>
      <div><span style='{label_style}'>Affidavit income</span><br>{money(fields.get("i_aff") or income_total, currency)}</div>
      <div><span style='{label_style}'>Tax verified income</span><br>{money(fields.get("i_tax"), currency)}</div>
    </div>
  </div>
"""

    if docs:
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06)'>"
            f"<span style='color:rgba(255,255,255,0.8)'>{doc.get('type','unknown').replace('_',' ').title()}</span>"
            f"<span style='color:rgba(255,255,255,0.45)'>{doc.get('pages',0)}p · {doc.get('quality','unknown')} · {badge(doc.get('status','seen'), 'pass')}</span></div>"
            for doc in docs
        )
        html += card("Documents", rows)

    if fields.get("name_variants"):
        rows = "".join(
            f"<div style='font-size:12px;color:rgba(255,255,255,0.7)'><span style='color:rgba(255,255,255,0.4)'>{key.replace('_',' ').title()}</span>: <span style='color:#e2e8f0'>{value}</span></div>"
            for key, value in fields["name_variants"].items()
        )
        html += card("Name consistency", rows)

    if findings:
        rows = "".join(
            f"<div style='padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06)'>"
            f"<div><b style='color:#e2e8f0'>{finding.get('rule_id','rule').replace('_',' ').title()}</b> {badge(finding.get('severity','info').title(), finding.get('severity','info'))}</div>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.55);margin-top:3px'>{finding.get('message','')}</div>"
            f"</div>"
            for finding in findings
        )
        html += card("Agent findings", rows)

    if synthesis.get("compound_flags"):
        rows = "".join(f"<div style='font-size:12px;padding:4px 0;color:rgba(255,255,255,0.75)'>{item}</div>" for item in synthesis["compound_flags"])
        html += card("Cross-document risks", rows, bg="rgba(245,158,11,0.08)", border="rgba(245,158,11,0.2)")

    adversarial = result.get("adversarial_audit", {})
    if adversarial.get("rejection_case") or adversarial.get("rebuttal_case"):
        rows = ""
        if adversarial.get("rejection_case"):
            rows += f"<div style='font-size:12px;padding:4px 0;color:rgba(255,255,255,0.75)'><b>Officer rejection case:</b> {adversarial['rejection_case']}</div>"
        if adversarial.get("rebuttal_case"):
            rows += f"<div style='font-size:12px;padding:4px 0;color:rgba(255,255,255,0.75)'><b>Applicant rebuttal path:</b> {adversarial['rebuttal_case']}</div>"
        html += card("Adversarial audit", rows)

    if result.get("next_steps"):
        rows = "".join(
            f"<div style='display:flex;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);align-items:flex-start'>{badge(step.get('priority','info').upper(), step.get('priority','info'))}<span style='color:rgba(255,255,255,0.75)'>{step.get('action','')}</span></div>"
            for step in result["next_steps"]
        )
        html += card("Next steps", rows)

    cert = result.get("deletion_cert") or raw_result.get("full_result", {}).get("deletion_cert")
    if cert:
        html += f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;margin-bottom:4px'><div style='font-size:11px;color:rgba(255,255,255,0.4)'>Deletion certificate</div><pre style='white-space:pre-wrap;font-size:10px;color:rgba(255,255,255,0.55);margin-top:6px'>{cert}</pre></div>"

    html += "</div>"
    return html


def card(title: str, body: str, bg: str = "rgba(255,255,255,0.04)", border: str = "rgba(255,255,255,0.08)") -> str:
    return f"<div style='background:{bg};border:1px solid {border};border-radius:10px;padding:16px;margin-bottom:12px;backdrop-filter:blur(8px)'><b style='color:#e2e8f0'>{title}</b><div style='margin-top:9px'>{body}</div></div>"


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
  <p>Immigration document intelligence — reads financial packets, audits evidence quality, flags inconsistencies, and builds a readiness roadmap.</p>
</div>
"""


CHECKLIST_HTML = """
<div class='tool-card' style='font-size:13px;line-height:1.9'>
  <b>📋 Packet checklist</b><br>
  <span style='color:#f87171'>Required:</span> affidavit, bank statement, tax return<br>
  <span style='color:#fbbf24'>Recommended:</span> bank balance certificate<br>
  <span style='color:#60a5fa'>Optional:</span> passport / identity document
  <hr>
  <b>🤖 Agent tools</b><br>
  <span style='color:rgba(255,255,255,0.6)'>Auditor:</span> document readiness and quality checks<br>
  <span style='color:rgba(255,255,255,0.6)'>Strategist:</span> roadmap and missing-evidence checklist<br>
  <span style='color:rgba(255,255,255,0.6)'>Synthesis:</span> cross-document risk narrative
</div>
"""


def build_ui():
    with gr.Blocks(
        title="Uplan - Immigration Document Intelligence",
    ) as app:
        result_state = gr.State({})
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=270):
                gr.Markdown("### Agent tools")
                demo_toggle = gr.Checkbox(label="Demo mode", value=True, info="Uses a clearly marked sample analysis. Turn off for live AMD backend.")
                upload_box = gr.File(label="Upload PDF packet", file_types=[".pdf"], file_count="multiple", visible=False)
                demo_toggle.change(lambda enabled: gr.update(visible=not enabled), demo_toggle, upload_box)
                analyse_btn = gr.Button("Analyse packet", variant="primary")
                gr.HTML(CHECKLIST_HTML)
                gr.Markdown("Raw documents are processed outside the UI in live mode. The extraction layer emits a deletion certificate after purge.")

            with gr.Column(scale=2, min_width=360):
                gr.Markdown("### Consultant")
                chatbot = gr.Chatbot(
                    label="Consultant",
                    height=560,
                    show_label=False,
                    value=[{"role": "assistant", "content": "Welcome to Uplan. Load the sample demo or upload a packet when the AMD backend is online. The dashboard will show whether the result is live or backend-offline."}],
                )
                with gr.Row():
                    chat_input = gr.Textbox(placeholder="Ask about risks, missing evidence, or next steps...", show_label=False, container=False, scale=5)
                    send_btn = gr.Button("Send", variant="secondary", scale=1)

            with gr.Column(scale=2, min_width=380):
                gr.Markdown("### Readiness dashboard")
                dashboard = gr.HTML("<p style='color:rgba(255,255,255,0.4);padding:16px;text-align:center'>Load demo or upload documents to begin.</p>")

        analyse_btn.click(handle_upload, [upload_box, demo_toggle], [dashboard, chatbot, result_state])
        send_btn.click(chat_response, [chat_input, chatbot, result_state], [chatbot, chat_input])
        chat_input.submit(chat_response, [chat_input, chatbot, result_state], [chatbot, chat_input])

    return app


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", font=["Georgia", "Times New Roman", "serif"]),
        css=CSS,
    )
