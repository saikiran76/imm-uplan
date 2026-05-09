"""
uplan.extraction.parser
-----------------------
Converts raw JSON dicts from the VLM into typed PageExtractionResult objects.

This is the boundary between "text the VLM produced" and
"typed data the rule engine trusts". All casting, range checking,
and confidence assignment happens here.

Confidence assignment logic
---------------------------
Confidence is determined by TWO independent signals, and the WORSE of the
two governs the final assignment:

  Signal 1 — VLM logprob (passed in as mean_logprob):
    < -0.30  → LOW
    -0.30 to -0.15 → MEDIUM
    > -0.15  → HIGH

  Signal 2 — Field-level confidence string the VLM returned:
    "low"    → LOW
    "medium" → MEDIUM
    "high"   → HIGH

The CI (confidence interval) is a heuristic ±% of the point estimate:
    HIGH   → ±2%
    MEDIUM → ±8%
    LOW    → ±20%

These are conservative defaults. In a production calibration pass,
these would be fit against a labelled ground-truth document set.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from .models import (
        AssetItem,
        Confidence,
        ConfidenceWrapper,
        FamilyMember,
        FinancialAccount,
        IncomeSource,
        PageExtractionResult,
        PageType,
        PropertyAsset,
        SourceQuality,
    )
except ImportError:
    from models import (
        AssetItem,
        Confidence,
        ConfidenceWrapper,
        FamilyMember,
        FinancialAccount,
        IncomeSource,
        PageExtractionResult,
        PageType,
        PropertyAsset,
        SourceQuality,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confidence assignment
# ─────────────────────────────────────────────────────────────────────────────

_CI_PCT: dict[Confidence, float] = {
    Confidence.HIGH:   0.02,
    Confidence.MEDIUM: 0.08,
    Confidence.LOW:    0.20,
}

def _logprob_to_confidence(lp: float) -> Confidence:
    if lp > -0.15:
        return Confidence.HIGH
    if lp > -0.30:
        return Confidence.MEDIUM
    return Confidence.LOW

def _string_to_confidence(s: str) -> Confidence:
    s = (s or "").lower()
    if s == "high":   return Confidence.HIGH
    if s == "medium": return Confidence.MEDIUM
    return Confidence.LOW

def _worst(a: Confidence, b: Confidence) -> Confidence:
    rank = {Confidence.HIGH: 2, Confidence.MEDIUM: 1, Confidence.LOW: 0}
    return a if rank[a] <= rank[b] else b

def _wrap(
    value: Any,
    field_conf_str: str,
    mean_logprob: float,
    source_quality: SourceQuality,
    raw_text: str = "",
) -> Optional[ConfidenceWrapper]:
    """
    Build a ConfidenceWrapper from a raw value and two confidence signals.
    Returns None if value cannot be cast to float.
    """
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None

    conf_lp   = _logprob_to_confidence(mean_logprob)
    conf_str  = _string_to_confidence(field_conf_str)
    final_conf = _worst(conf_lp, conf_str)

    ci_pct = _CI_PCT[final_conf]
    return ConfidenceWrapper(
        value=fval,
        confidence=final_conf,
        ci_low=fval * (1 - ci_pct),
        ci_high=fval * (1 + ci_pct),
        source_quality=source_quality,
        raw_text=raw_text,
    )


def _wrap_money_obj(
    data: Any,
    mean_logprob: float,
    source_quality: SourceQuality,
) -> Optional[ConfidenceWrapper]:
    if not isinstance(data, dict):
        return None
    value = data.get("value")
    return _wrap(
        value=value,
        field_conf_str=data.get("confidence", "low"),
        mean_logprob=mean_logprob,
        source_quality=source_quality,
        raw_text=str(value or ""),
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionParser:
    """
    Converts raw VLM output dicts into typed PageExtractionResult objects.
    One method per page type — the extractor calls the right one after
    page classification.
    """

    def parse_page(
        self,
        data: dict,
        page_number: int,
        page_type: PageType,
        mean_logprob: float,
    ) -> PageExtractionResult:
        """Dispatch to the correct parse method based on page_type."""
        quality_str = data.get("source_quality", "digital")
        try:
            quality = SourceQuality(quality_str)
        except ValueError:
            quality = SourceQuality.DIGITAL

        result = PageExtractionResult(
            page_number=page_number,
            page_type=page_type,
            source_quality=quality,
        )

        dispatch = {
            PageType.BANK_STATEMENT: self._parse_bank_statement,
            PageType.BANK_BALANCE_CERTIFICATE: self._parse_bank_balance_certificate,
            PageType.TAX_RETURN:     self._parse_tax_return,
            PageType.AFFIDAVIT:      self._parse_affidavit,
            PageType.EMPLOYMENT:     self._parse_application_form,
            PageType.IDENTITY:       self._parse_identity,
            PageType.SPONSOR:        self._parse_sponsor,
        }
        parse_fn = dispatch.get(page_type)
        if parse_fn:
            parse_fn(data, result, mean_logprob, quality)

        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _parse_bank_statement(
        self,
        d: dict,
        r: PageExtractionResult,
        lp: float,
        q: SourceQuality,
    ) -> None:
        r.currency_code  = d.get("currency_code")
        r.name_string    = d.get("account_holder_name")

        # Monthly closing balances
        for entry in d.get("monthly_closing_balances", []):
            w = _wrap(
                value=entry.get("closing_balance"),
                field_conf_str=entry.get("confidence", "low"),
                mean_logprob=lp,
                source_quality=q,
                raw_text=str(entry.get("closing_balance", "")),
            )
            if w:
                r.balance_series.append(w)

        # Deposit entries
        # month_offset is computed relative to the first month of the series.
        # Here we use a simple index since we don't have the period start parsed.
        # The merger will sort by date when full date parsing is implemented.
        for i, dep in enumerate(d.get("deposits", [])):
            w = _wrap(
                value=dep.get("amount"),
                field_conf_str=dep.get("confidence", "low"),
                mean_logprob=lp,
                source_quality=q,
                raw_text=str(dep.get("amount", "")),
            )
            if w:
                r.deposit_entries.append((float(i), w))

    def _parse_bank_balance_certificate(
        self,
        d: dict,
        r: PageExtractionResult,
        lp: float,
        q: SourceQuality,
    ) -> None:
        r.currency_code = d.get("currency_code")
        r.name_string = d.get("account_holder_name")
        r.financial_accounts.append(FinancialAccount(
            institution_name=d.get("institution_name"),
            account_number=str(d.get("account_number")) if d.get("account_number") is not None else None,
            account_type=d.get("account_type") or "balance_certificate",
            amount=_wrap_money_obj(d.get("available_balance"), lp, q),
        ))

    def _parse_tax_return(
        self,
        d: dict,
        r: PageExtractionResult,
        lp: float,
        q: SourceQuality,
    ) -> None:
        r.name_string = d.get("taxpayer_name")
        r.currency_code = d.get("currency_code")

        try:
            r.tax_year = int(d.get("tax_year", 0)) or None
        except (TypeError, ValueError):
            r.tax_year = None

        # Prefer taxable_income over gross_income as the canonical i_tax
        taxable = d.get("taxable_income", {}) or {}
        gross   = d.get("gross_income", {}) or {}

        r.i_tax = _wrap(
            value=taxable.get("value") or gross.get("value"),
            field_conf_str=taxable.get("confidence") or gross.get("confidence", "low"),
            mean_logprob=lp,
            source_quality=q,
        )

    def _parse_affidavit(
        self,
        d: dict,
        r: PageExtractionResult,
        lp: float,
        q: SourceQuality,
    ) -> None:
        r.name_string   = d.get("declarant_name")
        r.beneficiary_name = d.get("beneficiary_name")
        r.declarant_address = d.get("declarant_address")
        r.currency_code = d.get("currency_code")
        r.affidavit_type = d.get("affidavit_type")
        r.spon_relationship = d.get("relationship")

        income_d = d.get("declared_annual_income", {}) or {}
        r.i_aff = _wrap(
            value=income_d.get("value"),
            field_conf_str=income_d.get("confidence", "low"),
            mean_logprob=lp,
            source_quality=q,
        )

        for member in _as_list(d.get("family_members")):
            if not isinstance(member, dict):
                continue
            r.family_members.append(FamilyMember(
                name=member.get("name"),
                relationship=member.get("relationship"),
                date_of_birth=member.get("date_of_birth"),
                age=_as_optional_float(member.get("age")),
            ))

        for account in _as_list(d.get("financial_accounts")):
            if not isinstance(account, dict):
                continue
            r.financial_accounts.append(FinancialAccount(
                institution_name=account.get("institution_name"),
                account_number=str(account.get("account_number")) if account.get("account_number") is not None else None,
                account_type=account.get("account_type"),
                amount=_wrap_money_obj(account.get("amount"), lp, q),
            ))

        for income in _as_list(d.get("income_sources")):
            if not isinstance(income, dict):
                continue
            r.income_sources.append(IncomeSource(
                source=income.get("source"),
                annual_amount=_wrap_money_obj(income.get("annual_amount"), lp, q),
            ))

        for asset in _as_list(d.get("movable_assets")):
            if not isinstance(asset, dict):
                continue
            r.movable_assets.append(AssetItem(
                description=asset.get("description"),
                owner=asset.get("owner"),
                amount=_wrap_money_obj(asset.get("amount"), lp, q),
            ))

        for prop in _as_list(d.get("properties")):
            if not isinstance(prop, dict):
                continue
            r.properties.append(PropertyAsset(
                description=prop.get("description"),
                value=_wrap_money_obj(prop.get("value"), lp, q),
            ))

    def _parse_application_form(
        self,
        d: dict,
        r: PageExtractionResult,
        lp: float,
        q: SourceQuality,
    ) -> None:
        r.name_string   = d.get("applicant_name")
        r.currency_code = d.get("currency_code")

        income_d = d.get("declared_income", {}) or {}
        r.i_form = _wrap(
            value=income_d.get("value"),
            field_conf_str=income_d.get("confidence", "low"),
            mean_logprob=lp,
            source_quality=q,
        )

    def _parse_identity(
        self,
        d: dict,
        r: PageExtractionResult,
        lp: float,
        q: SourceQuality,
    ) -> None:
        # Prefer romanised name if original is non-Latin
        r.name_string = d.get("name_romanised") or d.get("full_name")
        r.doc_type = d.get("document_type")

    def _parse_sponsor(
        self,
        d: dict,
        r: PageExtractionResult,
        lp: float,
        q: SourceQuality,
    ) -> None:
        r.name_string       = d.get("sponsor_name")
        r.currency_code     = d.get("currency_code")
        r.spon_relationship = d.get("relationship")

        income_d = d.get("sponsor_annual_income", {}) or {}
        r.i_spon = _wrap(
            value=income_d.get("value"),
            field_conf_str=income_d.get("confidence", "low"),
            mean_logprob=lp,
            source_quality=q,
        )
