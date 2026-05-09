"""
Prompt templates for the page-based extraction pipeline.
"""

from __future__ import annotations

try:
    from .models import PageType
except ImportError:
    from models import PageType


SYSTEM_PROMPT = """You are a specialist financial document extraction engine.
Your sole purpose is to read the document image provided and return a single,
valid JSON object containing the specific fields requested.

Rules you must follow without exception:
- Output ONLY the JSON object. No explanation, no preamble, no markdown fences.
- If a field is not present in the document or is not legible, set it to null.
- Do not infer, estimate, or hallucinate any value. Read what is printed.
- All monetary amounts must be returned as numbers, not strings.
- Currency codes must be ISO 4217 when visible.
- Dates must use ISO 8601 where possible.
- Confidence values must be exactly one of "high", "medium", or "low".
- Names must be copied exactly as printed."""


PAGE_CLASSIFIER_PROMPT = """\
Classify this page and return ONLY valid JSON in this exact shape:
{
  "page_type": "<page type>",
  "language": "<language code or null>",
  "source_quality": "<digital|scan|photograph|unknown>",
  "confidence": "<high|medium|low>"
}

Valid page_type values:
- "bank_statement"
- "bank_balance_certificate"
- "tax_return"
- "affidavit"
- "sponsor_letter"
- "employment_letter"
- "identity_document"
- "unknown"
"""


CONTINUATION_CLASSIFIER_PROMPT = """\
Classify this page and return ONLY valid JSON in this exact shape:
{
  "page_type": "<page type>",
  "language": "<language code or null>",
  "source_quality": "<digital|scan|photograph|unknown>",
  "confidence": "<high|medium|low>"
}

The previous recognised page was "__PREVIOUS_PAGE_TYPE__".
This page may be a continuation of that document. If the page contains similar
names, financial tables, signatures, seals, address blocks, income statements,
family declarations, account tables, or continuation text, classify it as
"__PREVIOUS_PAGE_TYPE__". Otherwise classify it independently.

Valid page_type values:
- "bank_statement"
- "bank_balance_certificate"
- "tax_return"
- "affidavit"
- "sponsor_letter"
- "employment_letter"
- "identity_document"
- "unknown"
"""


BANK_STATEMENT_PROMPT = """\
Extract the bank statement fields from this page and return ONLY valid JSON:
{
  "account_holder_name": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "statement_period_start": "<YYYY-MM or null>",
  "statement_period_end": "<YYYY-MM or null>",
  "opening_balance": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "closing_balance": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "monthly_closing_balances": [
    {"month": "<YYYY-MM>", "closing_balance": <number>, "confidence": "<high|medium|low>"}
  ],
  "deposits": [
    {"date": "<YYYY-MM-DD or null>", "amount": <number>, "description": "<string or null>", "confidence": "<high|medium|low>"}
  ],
  "source_quality": "<digital|scan|photograph|unknown>"
}
"""


BANK_BALANCE_CERTIFICATE_PROMPT = """\
Extract the bank balance certificate fields from this page and return ONLY valid JSON:
{
  "account_holder_name": "<string or null>",
  "institution_name": "<string or null>",
  "account_number": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "certificate_date": "<YYYY-MM-DD or null>",
  "balances": [
    {
      "account_type": "<savings|fixed_deposit|term_deposit|current|other>",
      "amount": {"value": <number or null>, "confidence": "<high|medium|low>"},
      "description": "<string or null>"
    }
  ],
  "available_balance": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "exchange_rate": {"value": <number or null>, "currency_pair": "<string or null>", "confidence": "<high|medium|low>"},
  "held_minimum_days": <integer or null>,
  "source_quality": "<digital|scan|photograph|unknown>"
}

If the page is a bank certificate, bank balance certificate, account balance
letter, solvency certificate, or fixed-deposit balance certificate, use this
schema. Do not extract account numbers as monetary amounts.

Important:
- Extract every separately listed balance. If a sentence says "Rs. 245582.83 in
  savings and 1023668.40 in fixed deposits", return TWO balances:
  savings = 245582.83 and fixed_deposit = 1023668.40.
- available_balance should be the printed total only if a total is explicitly
  printed. Do not invent a total.
- If the certificate says balances were held for a minimum 90 day period, set
  held_minimum_days to 90.
"""


TAX_RETURN_PROMPT = """\
Extract the tax return fields from this page and return ONLY valid JSON:
{
  "taxpayer_name": "<string or null>",
  "tax_year": <integer or null>,
  "currency_code": "<ISO 4217 code or null>",
  "gross_income": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "taxable_income": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "tax_paid": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "income_sources": ["<string>", "..."],
  "document_issuer": "<string or null>",
  "source_quality": "<digital|scan|photograph|unknown>"
}
"""


AFFIDAVIT_PROMPT = """\
Extract the affidavit fields from this page and return ONLY valid JSON:
{
  "declarant_name": "<string or null>",
  "beneficiary_name": "<string or null>",
  "relationship": "<string or null>",
  "declarant_address": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "declared_annual_income": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "declared_net_worth": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "declared_support_amount": {"value": <number or null>, "confidence": "<high|medium|low>", "period": "<string or null>"},
  "family_members": [
    {"name": "<string or null>", "relationship": "<string or null>", "date_of_birth": "<YYYY-MM-DD or null>", "age": <number or null>}
  ],
  "financial_accounts": [
    {
      "institution_name": "<string or null>",
      "account_number": "<string or null>",
      "account_type": "<string or null>",
      "amount": {"value": <number or null>, "confidence": "<high|medium|low>"}
    }
  ],
  "income_sources": [
    {"source": "<salary|business|rent|other>", "annual_amount": {"value": <number or null>, "confidence": "<high|medium|low>"}}
  ],
  "movable_assets": [
    {"description": "<string or null>", "owner": "<string or null>", "amount": {"value": <number or null>, "confidence": "<high|medium|low>"}}
  ],
  "properties": [
    {"description": "<string or null>", "value": {"value": <number or null>, "confidence": "<high|medium|low>"}}
  ],
  "affidavit_type": "<string or null>",
  "notarised": <true|false|null>,
  "source_quality": "<digital|scan|photograph|unknown>"
}

For Indian number formatting, convert values exactly:
- Rs. 1,54,603.98 means 154603.98.
- Rs. 10.23,668-40 means 1023668.40 if the visible intent is Indian comma/decimal formatting.
- Do not turn account numbers into amounts.
- Do not invent totals that are not printed.
"""


SPONSOR_PROMPT = """\
Extract the sponsor letter fields from this page and return ONLY valid JSON:
{
  "sponsor_name": "<string or null>",
  "relationship": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "sponsor_annual_income": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "source_quality": "<digital|scan|photograph|unknown>"
}
"""


EMPLOYMENT_PROMPT = """\
Extract the employment or application form fields from this page and return ONLY valid JSON:
{
  "applicant_name": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "declared_income": {"value": <number or null>, "confidence": "<high|medium|low>"},
  "source_quality": "<digital|scan|photograph|unknown>"
}
"""


IDENTITY_PROMPT = """\
Extract the identity document fields from this page and return ONLY valid JSON:
{
  "document_type": "<string or null>",
  "full_name": "<string or null>",
  "name_romanised": "<string or null>",
  "source_quality": "<digital|scan|photograph|unknown>"
}
"""


_PROMPTS: dict[PageType, str] = {
    PageType.BANK_STATEMENT: BANK_STATEMENT_PROMPT,
    PageType.BANK_BALANCE_CERTIFICATE: BANK_BALANCE_CERTIFICATE_PROMPT,
    PageType.TAX_RETURN: TAX_RETURN_PROMPT,
    PageType.AFFIDAVIT: AFFIDAVIT_PROMPT,
    PageType.SPONSOR: SPONSOR_PROMPT,
    PageType.EMPLOYMENT: EMPLOYMENT_PROMPT,
    PageType.IDENTITY: IDENTITY_PROMPT,
}


def get_prompt(page_type: PageType) -> str:
    return _PROMPTS.get(page_type, "")
