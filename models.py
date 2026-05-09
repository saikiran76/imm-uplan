"""
Typed data structures for the page-based extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PageType(str, Enum):
    BANK_STATEMENT = "bank_statement"
    BANK_BALANCE_CERTIFICATE = "bank_balance_certificate"
    TAX_RETURN = "tax_return"
    AFFIDAVIT = "affidavit"
    SPONSOR = "sponsor_letter"
    EMPLOYMENT = "employment_letter"
    IDENTITY = "identity_document"
    UNKNOWN = "unknown"


class SourceQuality(str, Enum):
    DIGITAL = "digital"
    SCAN = "scan"
    PHOTOGRAPH = "photograph"
    UNKNOWN = "unknown"


@dataclass
class ConfidenceWrapper:
    value: float
    confidence: Confidence
    ci_low: float
    ci_high: float
    source_quality: SourceQuality
    raw_text: str = ""

    def is_reliable(self) -> bool:
        return self.confidence in {Confidence.HIGH, Confidence.MEDIUM}

    def midpoint(self) -> float:
        return (self.ci_low + self.ci_high) / 2.0


@dataclass
class FamilyMember:
    name: Optional[str] = None
    relationship: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[float] = None


@dataclass
class FinancialAccount:
    institution_name: Optional[str] = None
    account_number: Optional[str] = None
    account_type: Optional[str] = None
    amount: Optional[ConfidenceWrapper] = None


@dataclass
class IncomeSource:
    source: Optional[str] = None
    annual_amount: Optional[ConfidenceWrapper] = None


@dataclass
class AssetItem:
    description: Optional[str] = None
    owner: Optional[str] = None
    amount: Optional[ConfidenceWrapper] = None


@dataclass
class PropertyAsset:
    description: Optional[str] = None
    value: Optional[ConfidenceWrapper] = None


@dataclass
class PageExtractionResult:
    page_number: int
    page_type: PageType
    source_quality: SourceQuality

    balance_series: list[ConfidenceWrapper] = field(default_factory=list)
    deposit_entries: list[tuple[float, ConfidenceWrapper]] = field(default_factory=list)

    currency_code: Optional[str] = None
    i_tax: Optional[ConfidenceWrapper] = None
    i_form: Optional[ConfidenceWrapper] = None
    i_aff: Optional[ConfidenceWrapper] = None
    i_spon: Optional[ConfidenceWrapper] = None

    spon_relationship: Optional[str] = None
    tax_year: Optional[int] = None
    name_string: Optional[str] = None
    affidavit_type: Optional[str] = None
    doc_type: Optional[str] = None

    declarant_address: Optional[str] = None
    beneficiary_name: Optional[str] = None
    family_members: list[FamilyMember] = field(default_factory=list)
    financial_accounts: list[FinancialAccount] = field(default_factory=list)
    income_sources: list[IncomeSource] = field(default_factory=list)
    movable_assets: list[AssetItem] = field(default_factory=list)
    properties: list[PropertyAsset] = field(default_factory=list)


@dataclass
class DocumentExtractionResult:
    document_hash: str
    session_id: str
    deletion_cert: str
    filename: str
    total_pages: int
    source_quality: SourceQuality
    extraction_ms: int

    pages: list[PageExtractionResult] = field(default_factory=list)
    raw_purge_confirmed: bool = False

    balance_series: list[ConfidenceWrapper] = field(default_factory=list)
    deposit_entries: list[tuple[float, ConfidenceWrapper]] = field(default_factory=list)

    currency_code: Optional[str] = None
    i_tax: Optional[ConfidenceWrapper] = None
    i_form: Optional[ConfidenceWrapper] = None
    i_aff: Optional[ConfidenceWrapper] = None
    i_spon: Optional[ConfidenceWrapper] = None

    spon_relationship: Optional[str] = None
    tax_year: Optional[int] = None
    name_variants: dict[str, str] = field(default_factory=dict)

    declarant_address: Optional[str] = None
    beneficiary_name: Optional[str] = None
    family_members: list[FamilyMember] = field(default_factory=list)
    financial_accounts: list[FinancialAccount] = field(default_factory=list)
    income_sources: list[IncomeSource] = field(default_factory=list)
    movable_assets: list[AssetItem] = field(default_factory=list)
    properties: list[PropertyAsset] = field(default_factory=list)

    @property
    def doc_hash(self) -> str:
        return self.document_hash

    def _reliable_value(
        self,
        wrapper: Optional[ConfidenceWrapper],
    ) -> Optional[float]:
        if wrapper is None or not wrapper.is_reliable():
            return None
        return wrapper.value

    def _reliable_money(self, wrapper: Optional[ConfidenceWrapper]) -> Optional[float]:
        return self._reliable_value(wrapper)

    def reliable_fields(self) -> dict[str, object]:
        if not self.raw_purge_confirmed:
            raise RuntimeError(
                "PRIVACY GATE: reliable fields are unavailable until raw purge is confirmed."
            )

        latest_balance = self.balance_series[-1] if self.balance_series else None

        return {
            "i_tax": self._reliable_value(self.i_tax),
            "i_form": self._reliable_value(self.i_form),
            "i_aff": self._reliable_value(self.i_aff),
            "i_spon": self._reliable_value(self.i_spon),
            "balance_closing": self._reliable_value(latest_balance),
            "currency_code": self.currency_code,
            "tax_year": self.tax_year,
            "spon_relationship": self.spon_relationship,
            "name_variants": dict(self.name_variants),
            "declarant_address": self.declarant_address,
            "beneficiary_name": self.beneficiary_name,
            "family_members": [
                {
                    "name": member.name,
                    "relationship": member.relationship,
                    "date_of_birth": member.date_of_birth,
                    "age": member.age,
                }
                for member in self.family_members
            ],
            "financial_accounts": [
                {
                    "institution_name": account.institution_name,
                    "account_number": account.account_number,
                    "account_type": account.account_type,
                    "amount": self._reliable_money(account.amount),
                }
                for account in self.financial_accounts
            ],
            "income_sources": [
                {
                    "source": income.source,
                    "annual_amount": self._reliable_money(income.annual_amount),
                }
                for income in self.income_sources
            ],
            "movable_assets": [
                {
                    "description": asset.description,
                    "owner": asset.owner,
                    "amount": self._reliable_money(asset.amount),
                }
                for asset in self.movable_assets
            ],
            "properties": [
                {
                    "description": prop.description,
                    "value": self._reliable_money(prop.value),
                }
                for prop in self.properties
            ],
        }

    def any_low_confidence(self) -> bool:
        wrappers: list[ConfidenceWrapper] = []
        wrappers.extend(self.balance_series)
        wrappers.extend(wrapper for _, wrapper in self.deposit_entries)
        wrappers.extend(account.amount for account in self.financial_accounts if account.amount is not None)
        wrappers.extend(income.annual_amount for income in self.income_sources if income.annual_amount is not None)
        wrappers.extend(asset.amount for asset in self.movable_assets if asset.amount is not None)
        wrappers.extend(prop.value for prop in self.properties if prop.value is not None)
        for wrapper in (self.i_tax, self.i_form, self.i_aff, self.i_spon):
            if wrapper is not None:
                wrappers.append(wrapper)
        return any(wrapper.confidence == Confidence.LOW for wrapper in wrappers)

    def extraction_summary(self) -> dict[str, object]:
        return {
            "document_hash": self.document_hash,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "source_quality": self.source_quality.value,
            "extraction_ms": self.extraction_ms,
            "low_confidence": self.any_low_confidence(),
            "raw_purge_confirmed": self.raw_purge_confirmed,
        }
