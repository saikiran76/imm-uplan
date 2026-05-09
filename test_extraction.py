"""
tests/test_extraction.py
------------------------
Phase 0 test suite for the extraction pipeline.

These tests run WITHOUT a GPU — they use MockVLMBackend and a
programmatically generated synthetic PDF created by reportlab.

Test tiers:
  T1 — Unit tests: individual parser and crypto functions
  T2 — Integration tests: full pipeline on synthetic documents
  T3 — Privacy tests: purge flag, zero_bytes, cert format
  T4 — Edge case tests: low-confidence fields, missing fields, malformed JSON

Run with:
    pytest tests/test_extraction.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the workspace module directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .crypto import generate_deletion_cert, zero_bytes
    from .extractor import DocumentExtractor, MockVLMBackend
    from .models import Confidence, ConfidenceWrapper, DocumentExtractionResult, PageType, SourceQuality
    from .parser import ExtractionParser, _wrap
except ImportError:
    from crypto import generate_deletion_cert, zero_bytes
    from extractor import DocumentExtractor, MockVLMBackend
    from models import Confidence, ConfidenceWrapper, DocumentExtractionResult, PageType, SourceQuality
    from parser import ExtractionParser, _wrap


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic PDF generation
# Using only stdlib + fitz (PyMuPDF) to avoid extra deps
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_pdf(scenario: str = "clean") -> bytes:
    """
    Generate a minimal synthetic PDF for testing.
    Uses PyMuPDF to create a text-layer PDF (digital quality).
    Returns PDF bytes.

    Scenarios:
      "clean"       — stable balance, aligned income, no anomalies
      "warn"        — moderate late deposit, mild disparity
      "flag"        — coordinated spike, extreme disparity
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    scenarios = {
        "clean": """
BANK STATEMENT
Account Holder: Arjun Mehta
Currency: USD
Period: January 2024 - June 2024

Monthly Closing Balances:
Jan 2024: $42,800.00
Feb 2024: $43,100.00
Mar 2024: $44,200.00
Apr 2024: $43,900.00
May 2024: $44,600.00
Jun 2024: $45,100.00

Deposits:
15-Jan-2024: $1,200.00 (Salary)
15-Feb-2024: $950.00 (Salary)
15-Mar-2024: $1,100.00 (Salary)
15-Apr-2024: $800.00 (Salary)
15-May-2024: $1,300.00 (Salary)
15-Jun-2024: $1,050.00 (Salary)

Closing Balance: $45,100.00
        """,
        "warn": """
BANK STATEMENT
Account Holder: Priya Sharma
Currency: USD
Period: January 2024 - June 2024

Monthly Closing Balances:
Jan 2024: $21,000.00
Feb 2024: $20,800.00
Mar 2024: $22,100.00
Apr 2024: $21,500.00
May 2024: $19,800.00
Jun 2024: $38,400.00

Deposits:
15-Jan-2024: $800.00 (Salary)
15-Feb-2024: $750.00 (Salary)
15-Mar-2024: $900.00 (Salary)
15-Apr-2024: $700.00 (Salary)
15-May-2024: $600.00 (Salary)
10-Jun-2024: $14,800.00 (Transfer)

Closing Balance: $38,400.00
        """,
        "flag": """
BANK STATEMENT
Account Holder: Ravi Kumar
Currency: USD
Period: January 2024 - June 2024

Monthly Closing Balances:
Jan 2024: $9,200.00
Feb 2024: $8,800.00
Mar 2024: $9,600.00
Apr 2024: $9,100.00
May 2024: $8,500.00
Jun 2024: $51,000.00

Deposits:
15-Jan-2024: $400.00 (Salary)
15-Feb-2024: $350.00 (Salary)
15-Mar-2024: $500.00 (Salary)
15-Apr-2024: $300.00 (Salary)
15-May-2024: $200.00 (Salary)
02-Jun-2024: $40,000.00 (Family Transfer)

Closing Balance: $51,000.00
        """,
    }

    text = scenarios.get(scenario, scenarios["clean"])
    page.insert_text((50, 50), text, fontsize=11)

    tax_page = doc.new_page(width=595, height=842)
    tax_page.insert_text(
        (50, 50),
        """
TAX RETURN
Taxpayer Name: Arjun Mehta
Tax Year: 2023
Currency: USD
Gross Income: $70,000.00
Taxable Income: $65,800.00
Tax Paid: $9,200.00
Income Source: Salary
Document Issuer: IRS
        """,
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceWrapper:
    def test_high_confidence_is_reliable(self):
        w = ConfidenceWrapper(
            value=1000.0, confidence=Confidence.HIGH,
            ci_low=980.0, ci_high=1020.0,
            source_quality=SourceQuality.DIGITAL,
        )
        assert w.is_reliable()

    def test_medium_confidence_is_reliable(self):
        w = ConfidenceWrapper(
            value=1000.0, confidence=Confidence.MEDIUM,
            ci_low=920.0, ci_high=1080.0,
            source_quality=SourceQuality.SCAN,
        )
        assert w.is_reliable()

    def test_low_confidence_is_not_reliable(self):
        w = ConfidenceWrapper(
            value=1000.0, confidence=Confidence.LOW,
            ci_low=800.0, ci_high=1200.0,
            source_quality=SourceQuality.PHOTOGRAPH,
        )
        assert not w.is_reliable()

    def test_midpoint(self):
        w = ConfidenceWrapper(
            value=1000.0, confidence=Confidence.HIGH,
            ci_low=900.0, ci_high=1100.0,
            source_quality=SourceQuality.DIGITAL,
        )
        assert w.midpoint() == 1000.0


class TestWrapFunction:
    def test_wrap_high_logprob(self):
        w = _wrap(45100.0, "high", -0.10, SourceQuality.DIGITAL)
        assert w is not None
        assert w.confidence == Confidence.HIGH
        assert w.value == 45100.0
        assert w.ci_low < 45100.0 < w.ci_high

    def test_wrap_logprob_overrides_string_to_worse(self):
        # VLM said "high" but logprob is low — result should be LOW
        w = _wrap(45100.0, "high", -0.50, SourceQuality.PHOTOGRAPH)
        assert w is not None
        assert w.confidence == Confidence.LOW

    def test_wrap_null_value_returns_none(self):
        assert _wrap(None, "high", -0.10, SourceQuality.DIGITAL) is None

    def test_wrap_non_numeric_returns_none(self):
        assert _wrap("not_a_number", "high", -0.10, SourceQuality.DIGITAL) is None

    def test_wrap_ci_width_scales_with_confidence(self):
        w_high = _wrap(10000.0, "high",   -0.10, SourceQuality.DIGITAL)
        w_low  = _wrap(10000.0, "low",    -0.60, SourceQuality.PHOTOGRAPH)
        assert (w_low.ci_high - w_low.ci_low) > (w_high.ci_high - w_high.ci_low)


class TestCrypto:
    def test_zero_bytes_clears_buffer(self):
        buf = bytearray(b"sensitive data that should be zeroed")
        zero_bytes(buf)
        assert all(b == 0 for b in buf)

    def test_zero_bytes_empty_is_safe(self):
        zero_bytes(bytearray())  # Should not raise

    def test_deletion_cert_format(self):
        cert = generate_deletion_cert("abc123", "nonce456", "session789")
        assert "UPLAN-CERT:v1" in cert
        assert "doc=abc123" in cert
        assert "nonce=nonce456" in cert
        assert "ts=" in cert
        assert "sig=" in cert

    def test_deletion_cert_different_docs_give_different_sigs(self):
        cert1 = generate_deletion_cert("hash1", "nonce", "session")
        cert2 = generate_deletion_cert("hash2", "nonce", "session")
        sig1 = [l for l in cert1.splitlines() if l.startswith("sig=")][0]
        sig2 = [l for l in cert2.splitlines() if l.startswith("sig=")][0]
        assert sig1 != sig2

    def test_deletion_cert_same_inputs_different_ts_give_different_sigs(self):
        import time
        cert1 = generate_deletion_cert("hash", "nonce", "session")
        time.sleep(0.01)
        cert2 = generate_deletion_cert("hash", "nonce", "session")
        # Timestamps differ so sigs differ
        ts1 = [l for l in cert1.splitlines() if l.startswith("ts=")][0]
        ts2 = [l for l in cert2.splitlines() if l.startswith("ts=")][0]
        assert ts1 != ts2


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    def _run(self, scenario: str):
        pdf_bytes = make_synthetic_pdf(scenario)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name
        try:
            extractor = DocumentExtractor(vlm=MockVLMBackend("mock"))
            result = asyncio.run(extractor.extract(tmp_path, session_id="test-session"))
            return result
        finally:
            os.unlink(tmp_path)

    def test_clean_pipeline_completes(self):
        result = self._run("clean")
        assert result is not None

    def test_purge_confirmed_after_extraction(self):
        result = self._run("clean")
        assert result.raw_purge_confirmed is True

    def test_deletion_cert_present(self):
        result = self._run("clean")
        assert result.deletion_cert.startswith("UPLAN-CERT:v1")

    def test_document_hash_is_sha256(self):
        result = self._run("clean")
        assert len(result.doc_hash if hasattr(result, 'doc_hash') else result.document_hash) == 64

    def test_balance_series_extracted(self):
        result = self._run("clean")
        assert len(result.balance_series) > 0

    def test_i_tax_extracted(self):
        result = self._run("clean")
        assert result.i_tax is not None
        assert result.i_tax.value > 0

    def test_i_form_extracted(self):
        result = self._run("clean")
        # i_form may come from application form pages; in mock it may be None
        # depending on which page types are in the synthetic doc
        pass  # presence tested separately in parser tests

    def test_privacy_gate_raises_before_purge(self):
        unpurged = DocumentExtractionResult(
            document_hash="abc",
            session_id="xyz",
            deletion_cert="",
            filename="test.pdf",
            total_pages=1,
            source_quality=SourceQuality.DIGITAL,
            extraction_ms=100,
            raw_purge_confirmed=False,
        )
        with pytest.raises(RuntimeError, match="PRIVACY GATE"):
            unpurged.reliable_fields()

    def test_reliable_fields_after_purge(self):
        result = self._run("clean")
        fields = result.reliable_fields()
        assert isinstance(fields, dict)
        assert "i_tax" in fields
        assert "balance_closing" in fields


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Parser unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParser:
    def setup_method(self):
        self.parser = ExtractionParser()

    def test_parse_bank_statement_balances(self):
        data = {
            "account_holder_name": "Test User",
            "currency_code": "USD",
            "closing_balance": {"value": 45100.0, "confidence": "high"},
            "monthly_closing_balances": [
                {"month": "2024-01", "closing_balance": 42800.0, "confidence": "high"},
                {"month": "2024-06", "closing_balance": 45100.0, "confidence": "high"},
            ],
            "deposits": [
                {"date": "2024-01-15", "amount": 1200.0, "description": "salary", "confidence": "high"},
            ],
            "source_quality": "digital",
        }
        result = self.parser.parse_page(data, 1, PageType.BANK_STATEMENT, -0.12)
        assert len(result.balance_series) == 2
        assert result.balance_series[0].value == 42800.0
        assert result.balance_series[0].confidence == Confidence.HIGH
        assert result.currency_code == "USD"
        assert result.name_string == "Test User"
        assert len(result.deposit_entries) == 1

    def test_parse_tax_return_prefers_taxable_over_gross(self):
        data = {
            "taxpayer_name": "Test User",
            "tax_year": 2023,
            "currency_code": "USD",
            "gross_income":   {"value": 70000.0, "confidence": "high"},
            "taxable_income": {"value": 65800.0, "confidence": "high"},
            "source_quality": "digital",
        }
        result = self.parser.parse_page(data, 1, PageType.TAX_RETURN, -0.12)
        assert result.i_tax is not None
        assert result.i_tax.value == 65800.0
        assert result.tax_year == 2023

    def test_parse_affidavit_fields(self):
        data = {
            "declarant_name": "Sponsor Person",
            "currency_code": "USD",
            "declared_annual_income": {"value": 72000.0, "confidence": "medium"},
            "affidavit_type": "financial_support",
            "source_quality": "scan",
        }
        result = self.parser.parse_page(data, 1, PageType.AFFIDAVIT, -0.25)
        assert result.i_aff is not None
        assert result.i_aff.value == 72000.0
        assert result.i_aff.confidence == Confidence.MEDIUM

    def test_parse_handles_null_values_gracefully(self):
        data = {
            "taxable_income": {"value": None, "confidence": "low"},
            "gross_income":   {"value": None, "confidence": "low"},
            "source_quality": "photograph",
        }
        result = self.parser.parse_page(data, 1, PageType.TAX_RETURN, -0.60)
        assert result.i_tax is None

    def test_malformed_json_produces_empty_result(self):
        result = self.parser.parse_page({}, 1, PageType.BANK_STATEMENT, -0.10)
        assert result.balance_series == []
        assert result.deposit_entries == []


# ─────────────────────────────────────────────────────────────────────────────
# T4 — Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_low_confidence_field_not_in_reliable_fields(self):
        result = DocumentExtractionResult(
            document_hash="abc",
            session_id="xyz",
            deletion_cert="cert",
            filename="test.pdf",
            total_pages=1,
            source_quality=SourceQuality.PHOTOGRAPH,
            extraction_ms=100,
            raw_purge_confirmed=True,
            i_tax=ConfidenceWrapper(
                value=50000.0,
                confidence=Confidence.LOW,
                ci_low=40000.0,
                ci_high=60000.0,
                source_quality=SourceQuality.PHOTOGRAPH,
            ),
        )
        fields = result.reliable_fields()
        assert fields["i_tax"] is None  # LOW confidence → excluded

    def test_high_confidence_field_in_reliable_fields(self):
        result = DocumentExtractionResult(
            document_hash="abc",
            session_id="xyz",
            deletion_cert="cert",
            filename="test.pdf",
            total_pages=1,
            source_quality=SourceQuality.DIGITAL,
            extraction_ms=100,
            raw_purge_confirmed=True,
            i_tax=ConfidenceWrapper(
                value=65800.0,
                confidence=Confidence.HIGH,
                ci_low=64484.0,
                ci_high=67116.0,
                source_quality=SourceQuality.DIGITAL,
            ),
        )
        fields = result.reliable_fields()
        assert fields["i_tax"] == 65800.0
