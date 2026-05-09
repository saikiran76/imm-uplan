from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr


USE_LIVE_BACKEND = os.getenv("USE_LIVE_BACKEND", "false").lower() == "true"
AMD_ENDPOINT = os.getenv("AMD_ENDPOINT", "http://localhost:8000/analyze")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SAMPLE_PATH = Path(__file__).parent / "sample_outputs" / "demo_result.json"
DEMO_RESULT = json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else {}


CSS = """
.gradio-container { max-width: 1440px !important; }
#uplan-header {
  border-bottom: 1px solid #e4e4df;
  padding: 10px 0 16px;
  margin-bottom: 14px;
}
#uplan-header h1 {
  margin: 0;
  font-size: 24px;
  letter-spacing: 0;
}
#uplan-header p {
  margin: 5px 0 0;
  color: #666;
  font-size: 13px;
}
.tool-card {
  border: 1px solid #e1e1dc;
  border-radius: 8px;
  padding: 12px;
  background: #ffffff;
}
"""


SEV_COLOR = {
    "critical": ("#FCEBEB", "#A32D2D"),
    "warning": ("#FAEEDA", "#854F0B"),
    "info": ("#E6F1FB", "#0C447C"),
    "pass": ("#E1F5EE", "#085041"),
}


def badge(text: str, severity: str) -> str:
    bg, color = SEV_COLOR.get(severity, ("#F1EFE8", "#444441"))
    return (
        f"<span style='display:inline-block;padding:2px 9px;border-radius:999px;"
        f"font-size:11px;font-weight:600;background:{bg};color:{color};"
        f"margin:2px 3px 2px 0'>{text}</span>"
    )


def money(value: Any, currency: str = "") -> str:
    if value is None:
        return "<span style='color:#999'>not found</span>"
    try:
        return f"<b>{currency} {float(value):,.0f}</b>"
    except (TypeError, ValueError):
        return f"<b>{value}</b>"


def score_bar(score: float) -> str:
    pct = max(0, min(100, int(score * 100)))
    color = "#A32D2D" if pct < 50 else "#BA7517" if pct < 75 else "#1D9E75"
    return f"""
<div style='margin-top:8px'>
  <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px'>
    <span>Narrative coherence</span><b>{pct}%</b>
  </div>
  <div style='height:7px;background:#e8e8e3;border-radius:999px'>
    <div style='height:7px;width:{pct}%;background:{color};border-radius:999px'></div>
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
            "sponsor_name": fields.get("name_variants", {}).get("affidavit", "Sponsor"),
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
    if not USE_LIVE_BACKEND or not files:
        time.sleep(0.8)
        return DEMO_RESULT

    import requests

    upload_files = []
    handles = []
    try:
        for file_obj in files:
            path = getattr(file_obj, "name", file_obj)
            handle = open(path, "rb")
            handles.append(handle)
            upload_files.append(("files", (Path(path).name, handle, "application/pdf")))
        response = requests.post(AMD_ENDPOINT, files=upload_files, timeout=180)
        response.raise_for_status()
        return response.json()
    finally:
        for handle in handles:
            handle.close()


def build_dashboard_html(raw_result: dict[str, Any]) -> str:
    result = normalize_result(raw_result)
    if not result:
        return "<p style='color:#777;padding:16px'>Upload documents or run demo mode to begin.</p>"

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

    html = f"""
<div style='font-family:Inter,Arial,sans-serif;font-size:13px;line-height:1.55'>
  <div style='background:#f8f8f6;border:1px solid #e1e1dc;border-radius:8px;padding:14px;margin-bottom:12px'>
    <div style='display:flex;justify-content:space-between;align-items:center'>
      <b>Readiness overview</b>{badge(overall, overall_sev)}
    </div>
    {score_bar(score)}
    {"<div style='font-size:12px;color:#A32D2D;margin-top:7px'>Human review required</div>" if synthesis.get("human_review_required") else ""}
  </div>

  <div style='background:#fff;border:1px solid #e1e1dc;border-radius:8px;padding:14px;margin-bottom:12px'>
    <b>Parties</b>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:9px'>
      <div><span style='color:#777;font-size:11px'>Applicant</span><br><b>{result.get("applicant_name") or fields.get("beneficiary_name") or "not found"}</b></div>
      <div><span style='color:#777;font-size:11px'>Sponsor</span><br><b>{result.get("sponsor_name") or "not found"}</b></div>
    </div>
    <div style='margin-top:7px;color:#555'>Relationship: <b>{result.get("sponsor_relationship") or fields.get("spon_relationship") or "not found"}</b></div>
  </div>

  <div style='background:#fff;border:1px solid #e1e1dc;border-radius:8px;padding:14px;margin-bottom:12px'>
    <b>Financial summary</b>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:9px'>
      <div><span style='color:#777;font-size:11px'>Required funds</span><br>{money(t_req, currency)}</div>
      <div><span style='color:#777;font-size:11px'>Liquid / deposit evidence</span><br>{money(account_total or fields.get("balance_closing"), currency)}</div>
      <div><span style='color:#777;font-size:11px'>Affidavit income</span><br>{money(fields.get("i_aff") or income_total, currency)}</div>
      <div><span style='color:#777;font-size:11px'>Tax verified income</span><br>{money(fields.get("i_tax"), currency)}</div>
    </div>
  </div>
"""

    if docs:
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f1f1ed'>"
            f"<span>{doc.get('type','unknown').replace('_',' ').title()}</span>"
            f"<span style='color:#777'>{doc.get('pages',0)}p · {doc.get('quality','unknown')} · {badge(doc.get('status','seen'), 'pass')}</span></div>"
            for doc in docs
        )
        html += card("Documents", rows)

    if fields.get("name_variants"):
        rows = "".join(
            f"<div style='font-size:12px'><span style='color:#777'>{key.replace('_',' ').title()}</span>: {value}</div>"
            for key, value in fields["name_variants"].items()
        )
        html += card("Name consistency", rows)

    if findings:
        rows = "".join(
            f"<div style='padding:8px 0;border-bottom:1px solid #f1f1ed'>"
            f"<div><b>{finding.get('rule_id','rule').replace('_',' ').title()}</b> {badge(finding.get('severity','info').title(), finding.get('severity','info'))}</div>"
            f"<div style='font-size:12px;color:#555'>{finding.get('message','')}</div>"
            f"</div>"
            for finding in findings
        )
        html += card("Agent findings", rows)

    if synthesis.get("compound_flags"):
        rows = "".join(f"<div style='font-size:12px;padding:4px 0'>{item}</div>" for item in synthesis["compound_flags"])
        html += card("Cross-document risks", rows, bg="#FAEEDA", border="#E7C27A")

    adversarial = result.get("adversarial_audit", {})
    if adversarial.get("rejection_case") or adversarial.get("rebuttal_case"):
        rows = ""
        if adversarial.get("rejection_case"):
            rows += f"<div style='font-size:12px;padding:4px 0'><b>Officer rejection case:</b> {adversarial['rejection_case']}</div>"
        if adversarial.get("rebuttal_case"):
            rows += f"<div style='font-size:12px;padding:4px 0'><b>Applicant rebuttal path:</b> {adversarial['rebuttal_case']}</div>"
        html += card("Adversarial audit", rows)

    if result.get("next_steps"):
        rows = "".join(
            f"<div style='display:flex;gap:8px;padding:6px 0;border-bottom:1px solid #f1f1ed'>{badge(step.get('priority','info').upper(), step.get('priority','info'))}<span>{step.get('action','')}</span></div>"
            for step in result["next_steps"]
        )
        html += card("Next steps", rows)

    cert = result.get("deletion_cert") or raw_result.get("full_result", {}).get("deletion_cert")
    if cert:
        html += f"<div style='background:#f4f4f1;border-radius:8px;padding:10px;margin-bottom:4px'><div style='font-size:11px;color:#777'>Deletion certificate</div><pre style='white-space:pre-wrap;font-size:10px;color:#444'>{cert}</pre></div>"

    html += "</div>"
    return html


def card(title: str, body: str, bg: str = "#fff", border: str = "#e1e1dc") -> str:
    return f"<div style='background:{bg};border:1px solid {border};border-radius:8px;padding:14px;margin-bottom:12px'><b>{title}</b><div style='margin-top:8px'>{body}</div></div>"


SYSTEM_PROMPT = """You are Uplan's immigration document consultant AI.

Use the supplied document analysis to explain risks, missing evidence, and next steps.
Do not give legal advice. Keep replies concise and direct.

Analysis:
{analysis}
"""


def chat_response(message: str, history: list, result_state: dict) -> tuple[list, str]:
    result = normalize_result(result_state)
    if not message:
        return history, ""

    if not ANTHROPIC_API_KEY:
        findings = result.get("agent_findings", [])
        first = findings[0]["message"] if findings else "Upload bank statements and tax returns to enable cross-document checks."
        reply = (
            "Consultant chat is in local demo mode because ANTHROPIC_API_KEY is not set. "
            f"The main issue I see is: {first} "
            "The dashboard on the right will update as richer packet analysis is loaded."
        )
    else:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            messages = []
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": message})
            response = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=500,
                system=SYSTEM_PROMPT.format(analysis=json.dumps(result, indent=2)[:7000]),
                messages=messages,
            )
            reply = response.content[0].text
        except Exception as exc:
            reply = f"Consultant chat is temporarily unavailable: {exc}"

    return history + [[message, reply]], ""


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
    history = [["Analysis complete. What are my main issues?", intro]]
    return build_dashboard_html(normalized), history, normalized


HEADER_HTML = """
<div id='uplan-header'>
  <h1>Uplan</h1>
  <p>Immigration document intelligence: reads financial packets, audits evidence quality, flags inconsistencies, and builds a readiness roadmap.</p>
</div>
"""


CHECKLIST_HTML = """
<div class='tool-card' style='font-size:13px;line-height:1.8'>
  <b>Packet checklist</b><br>
  <span style='color:#A32D2D'>Required:</span> affidavit, bank statement, tax return<br>
  <span style='color:#854F0B'>Recommended:</span> bank balance certificate<br>
  <span style='color:#0C447C'>Optional:</span> passport / identity document<br>
  <hr style='border:0;border-top:1px solid #eee'>
  <b>Agent tools</b><br>
  Auditor: document readiness and quality checks<br>
  Strategist: roadmap and missing-evidence checklist<br>
  Synthesis: cross-document risk narrative
</div>
"""


def build_ui():
    with gr.Blocks(
        title="Uplan - Immigration Document Intelligence",
        theme=gr.themes.Soft(primary_hue="blue", font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]),
        css=CSS,
    ) as app:
        result_state = gr.State({})
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=270):
                gr.Markdown("### Agent tools")
                demo_toggle = gr.Checkbox(label="Demo mode", value=True, info="Uses sample analysis for HF CPU Space.")
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
                    bubble_full_width=False,
                    value=[[None, "Welcome to Uplan. Load the demo or upload a packet, and I will walk through the agent findings."]],
                )
                with gr.Row():
                    chat_input = gr.Textbox(placeholder="Ask about risks, missing evidence, or next steps...", show_label=False, container=False, scale=5)
                    send_btn = gr.Button("Send", variant="secondary", scale=1)

            with gr.Column(scale=2, min_width=380):
                gr.Markdown("### Readiness dashboard")
                dashboard = gr.HTML("<p style='color:#777;padding:16px'>Load demo or upload documents to begin.</p>")

        analyse_btn.click(handle_upload, [upload_box, demo_toggle], [dashboard, chatbot, result_state])
        send_btn.click(chat_response, [chat_input, chatbot, result_state], [chatbot, chat_input])
        chat_input.submit(chat_response, [chat_input, chatbot, result_state], [chatbot, chat_input])

    return app


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
