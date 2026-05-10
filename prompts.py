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
- First, output a <scratchpad> block to map out raw values, normalize them, and perform any required addition.
- Then, output ONLY the JSON object. No explanation, no preamble, no markdown fences.
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
### SYSTEM DIRECTIVE: QWEN2-VL 72B FORENSIC EXTRACTION
You are analyzing high-stakes financial documentation for immigration intelligence. Precision is non-negotiable.

### STEP 1: <scratchpad> REASONING (DO NOT SKIP)
Before outputting any JSON, you must use a <scratchpad> block to perform a "Forensic Count" for the bank statement:
1. RAW STRING EXTRACTION: Extract the figures exactly as they appear.
2. NORMALIZATION: Strip all currency symbols, commas, and trailing paise/decimals.
3. LINEAR LEDGER:
   - Identify: Opening Balance, Total Inflows, Total Outflows, and Closing Balance.
   - Verify: Opening + Inflows - Outflows == Closing. Flag any discrepancy.

### STEP 2: STICKY CONTEXT RETENTION
For multi-page documents, do not truncate the audit. If the document is dense, prioritize the integrity of the closing JSON block. The reasoning must lead logically to the final integers.

### STEP 3: FLAT JSON SCHEMA
Output ONLY a flat JSON object after the </scratchpad>.

{
  "account_holder_name": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "statement_period_start": "<YYYY-MM or null>",
  "statement_period_end": "<YYYY-MM or null>",
  "financial_indicators": {
      "opening_balance": <integer or null>,
      "total_inflows": <integer or null>,
      "total_outflows": <integer or null>,
      "closing_balance": <integer or null>,
      "math_verified": <true|false|null>
  },
  "deposits": [
    {"date": "<YYYY-MM-DD or null>", "amount_inr": <integer>, "description": "<string or null>"}
  ],
  "adversarial_flags": ["<string>"]
}
"""


BANK_BALANCE_CERTIFICATE_PROMPT = """\
Extract the bank balance certificate fields from this page and return ONLY valid JSON:
{
  "account_holder_name": "<string or null>",
  "institution_name": "<string or null>",
  "account_number": "<string or null>",
  "beneficiary_name": "<string or null>",
  "relationship": "<string or null>",
  "purpose": "<string or null>",
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
- If the certificate says it was requested for a son/daughter/spouse/parent or
  for another named person's education/visa, extract that person's name as
  beneficiary_name and the relationship exactly as printed.
"""


TAX_RETURN_PROMPT = """\
### SYSTEM DIRECTIVE: QWEN2-VL 72B FORENSIC EXTRACTION
You are analyzing high-stakes financial documentation for immigration intelligence. Precision is non-negotiable.

### STEP 1: <scratchpad> REASONING (DO NOT SKIP)
Before outputting any JSON, you must use a <scratchpad> block to perform a "Forensic Count" for the tax return:
1. RAW STRING EXTRACTION: Extract the figures exactly as they appear.
2. NORMALIZATION: Strip all currency symbols, commas, and trailing paise/decimals.
3. INCOME AGGREGATION:
   - Explicitly list every income stream (e.g., Salary, Business, Capital Gains) individually.
   - Explicitly calculate the SUM of these streams to derive the Gross Total Income.
   - Identify the final Taxable Income.

### STEP 2: STICKY CONTEXT RETENTION
For multi-page documents, do not truncate the audit. If the document is dense, prioritize the integrity of the closing JSON block. The reasoning must lead logically to the final integers.

### STEP 3: FLAT JSON SCHEMA
Output ONLY a flat JSON object after the </scratchpad>.

{
  "taxpayer_name": "<string or null>",
  "currency_code": "<ISO 4217 code or null>",
  "tax_year": <integer or null>,
  "income_streams": [{"source": "<string or null>", "amount_inr": <integer or null>}],
  "financial_indicators": {
      "gross_total_income": <integer or null>,
      "taxable_income": <integer or null>,
      "math_verified": <true|false|null>
  },
  "adversarial_flags": ["<string>"]
}
"""


AFFIDAVIT_PROMPT = """\
### SYSTEM DIRECTIVE: QWEN2-VL 72B FORENSIC EXTRACTION
You are analyzing high-stakes financial documentation for immigration intelligence. Precision is non-negotiable.

### STEP 1: <scratchpad> REASONING (DO NOT SKIP)
Before outputting any JSON, you must use a <scratchpad> block to perform a "Forensic Count" for every financial figure:
1. RAW STRING EXTRACTION: Extract the figure exactly as it appears (e.g., "Rs. 25,00,00-00").
2. NORMALIZATION: 
   - Strip all currency symbols and letters.
   - Remove trailing paise/decimals (e.g., "-00", "/-").
   - Count the remaining digits to verify the magnitude (Lakhs vs. Thousands).
3. AGGREGATION (For multiple streams): 
   - List every income source (Salary, Business, Rent) individually.
   - Explicitly calculate the SUM of these sources in the scratchpad.
4. LINEAR LEDGER (For Bank Statements/High Volume):
   - Identify: Opening Balance, Total Inflows, Total Outflows, and Closing Balance.
   - Verify: Opening + Inflows - Outflows == Closing. Flag any discrepancy.

### STEP 2: STICKY CONTEXT RETENTION
For multi-page documents, do not truncate the audit. If the document is dense, prioritize the integrity of the closing JSON block. The reasoning must lead logically to the final integers.

### STEP 3: FLAT JSON SCHEMA
Output ONLY a flat JSON object after the </scratchpad>.

{
  "applicant_name": "<string or null>",
  "sponsor_name": "<string or null>",
  "movable_assets_inr": <integer or null>,
  "total_annual_income_inr": <integer or null>,
  "income_streams": [{"source": "<string or null>", "amount_inr": <integer or null>}],
  "financial_indicators": {
      "opening_balance": <integer or null>,
      "closing_balance": <integer or null>,
      "math_verified": <true|false|null>
  },
  "adversarial_flags": ["<string>"]
}
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
