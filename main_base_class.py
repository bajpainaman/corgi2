from __future__ import annotations
from typing   import List, Optional, Literal
from datetime import date, datetime
from pydantic import BaseModel, Field, condecimal, constr

# ──────────────── PAGE TEXT ─────────────────
class PageText(BaseModel):
    page_num: int
    text: str                # raw unicode, no markdown
    markdown: Optional[str]  # keep if you want Mistral's MD

# ──────────────── BASE ─────────────────
class BaseDoc(BaseModel):
    claim_id:       constr(strip_whitespace=True)
    file_name:      str
    doc_class:      Literal[
        "LEASE", "ADDENDUM", "LEDGER",
        "SDI", "MOVE_OUT_STATEMENT",
        "INVOICE", "COLLECTION_LETTER",
        "APPLICATION",
        "PHOTO_REPORT", "EMAIL_CORRESPONDENCE",
        "POLICY", "OTHER"
    ]
    ocr_confidence: condecimal(max_digits=4, decimal_places=2) = Field(..., ge=0, le=1)
    parse_ts:       datetime = Field(default_factory=datetime.utcnow)
    # NEW ↓ Strip PDF to the studs - capture every single glyph
    page_text:      Optional[List[PageText]] = None   # filled for ANY doc
    full_text:      Optional[str] = None              # concatenated for convenience
    file_size_bytes: int                              # file size tracking
    page_count:     Optional[int] = None              # actual page count

# ──────────────── LEASE / ADDENDUM ─────
class LeaseInfo(BaseModel):
    tenants:              List[str]
    landlord:             str
    property_address:     str
    lease_start:          date
    lease_end:            date
    base_rent:            condecimal(max_digits=10, decimal_places=2)
    deposit_held:         condecimal(max_digits=10, decimal_places=2)
    late_fee_amount:      Optional[condecimal(max_digits=7, decimal_places=2)]
    late_fee_grace_days:  Optional[int]
    early_termination_fee:Optional[condecimal(max_digits=7, decimal_places=2)]
    deposit_waiver:       Optional[bool]
    governing_law_state:  Optional[str]
    # ★ new
    renewal_option:       Optional[bool]
    rent_due_day:         Optional[int]
    pet_fee:              Optional[condecimal(max_digits=7, decimal_places=2)]
    pet_deposit:          Optional[condecimal(max_digits=7, decimal_places=2)]
    utilities_responsible:Optional[List[str]]
    occupants_limit:      Optional[int]
    cosigner_names:       Optional[List[str]]
    signatures:           Optional[List[str]]
    executed_date:        Optional[date]
    # ★ Deposit replacement/waiver fields
    waiver_premium:       Optional[condecimal(max_digits=7, decimal_places=2)]
    provider_name:        Optional[str]
    coverage_amount:      Optional[condecimal(max_digits=10, decimal_places=2)]
    # ★ Optional fields with heuristic extraction
    tenant_forwarding_address: Optional[str]
    page_count:           Optional[int]
    lease_term_months:    Optional[int]

class LeaseDoc(BaseDoc):
    doc_class: Literal["LEASE", "ADDENDUM"]
    lease:     LeaseInfo

# ──────────────── LEDGER ───────────────
class LedgerLine(BaseModel):
    tx_date:      date
    code:         str
    description:  str
    amount:       condecimal(max_digits=10, decimal_places=2)
    running_bal:  condecimal(max_digits=10, decimal_places=2)

class AgingBucket(BaseModel):
    bucket: Literal["0-30","31-60","61-90","91+"]
    amount: condecimal(max_digits=10, decimal_places=2)

class LedgerSummary(BaseModel):
    period_start:    date
    period_end:      date
    balance_due:     condecimal(max_digits=10, decimal_places=2)
    total_charges:   condecimal(max_digits=10, decimal_places=2)
    total_credits:   condecimal(max_digits=10, decimal_places=2)
    # ★ new
    last_payment_date:   Optional[date]
    last_payment_amount: Optional[condecimal(max_digits=10, decimal_places=2)]
    aging:               Optional[List[AgingBucket]]
    late_fee_total:      Optional[condecimal(max_digits=10, decimal_places=2)]

class LedgerDoc(BaseDoc):
    doc_class: Literal["LEDGER"]
    summary:   LedgerSummary
    lines:     List[LedgerLine]

# ──────────────── SDI / MOVE OUT ───────
class ChargeBreakdown(BaseModel):
    category: Literal[
        "RENT","CLEANING","DAMAGE","RELET_FEE","LATE_FEES",
        "UTILITIES","PEST","CARPET","PAINT","LOCKS","OTHER"
    ]
    amount:     condecimal(max_digits=10, decimal_places=2)
    description:Optional[str]

class SDIInfo(BaseModel):
    move_out_date:  date
    notice_sent:    Optional[date]
    deposit_held:   condecimal(max_digits=10, decimal_places=2)
    total_charges:  condecimal(max_digits=10, decimal_places=2)
    refund_due:     condecimal(max_digits=10, decimal_places=2)
    charges:        List[ChargeBreakdown]
    # ★ new
    statutory_deadline_days: Optional[int]
    sent_within_deadline:    Optional[bool]
    photos_evidence:         Optional[bool]
    # ★ Optional fields with heuristic extraction
    tenant_forwarding_address: Optional[str]
    page_count:              Optional[int]
    refund_check_number:     Optional[str]

class SDIDoc(BaseDoc):
    doc_class: Literal["SDI","MOVE_OUT_STATEMENT"]
    sdi:       SDIInfo

# ──────────────── INVOICE / BILL ───────
class InvoiceLine(BaseModel):
    description: str
    qty:         Optional[condecimal(max_digits=10, decimal_places=2)]
    rate:        Optional[condecimal(max_digits=10, decimal_places=2)]
    amount:      condecimal(max_digits=10, decimal_places=2)

class InvoiceInfo(BaseModel):
    invoice_number:  str
    vendor_name:     str
    vendor_phone:    Optional[str]
    vendor_email:    Optional[str]
    invoice_date:    date
    due_date:        Optional[date]
    po_number:       Optional[str]
    work_order_id:   Optional[str]
    line_items:      List[InvoiceLine]
    tax:             Optional[condecimal(max_digits=10, decimal_places=2)]
    total:           condecimal(max_digits=10, decimal_places=2)
    paid_date:       Optional[date]
    check_number:    Optional[str]
    # ★ Optional fields with heuristic extraction
    vendor_license_number: Optional[str]
    nsf_fee_amount:   Optional[condecimal(max_digits=7, decimal_places=2)]
    page_count:       Optional[int]

class InvoiceDoc(BaseDoc):
    doc_class: Literal["INVOICE"]
    invoice:   InvoiceInfo

# ──────────────── COLLECTION LETTER ────
class CollectionLetter(BaseDoc):
    doc_class:          Literal["COLLECTION_LETTER"]
    issue_date:         date
    debtor_names:       List[str]
    amount_due:         condecimal(max_digits=10, decimal_places=2)
    deadline_date:      Optional[date]
    agency_name:        Optional[str]
    threats_disclosed:  Optional[bool]
    ledger_balance_ref: Optional[condecimal(max_digits=10, decimal_places=2)]
    # ★ new
    dispute_instructions_present: Optional[bool]
    interest_rate_applied:        Optional[condecimal(max_digits=5, decimal_places=2)]
    attorney_letterhead:          Optional[bool]

# ──────────────── APPLICATION ──────────
class ApplicationDoc(BaseDoc):
    doc_class:        Literal["APPLICATION"]
    applicant_names:  List[str]
    ssn_last4:        Optional[str]
    dob:              Optional[date]
    income_monthly:   Optional[condecimal(max_digits=10, decimal_places=2)]
    employer:         Optional[str]
    credit_score:     Optional[int]
    # ★ new
    prior_landlord:   Optional[str]
    prior_rent:       Optional[condecimal(max_digits=10, decimal_places=2)]
    pets:             Optional[bool]
    vehicles:         Optional[int]

# ──────────────── PHOTO REPORT ─────────
class PhotoReportDoc(BaseDoc):
    doc_class: Literal["PHOTO_REPORT"]
    photo_count: int
    contains_before_photos: Optional[bool]
    contains_after_photos:  Optional[bool]
    link: Optional[str]
    # ★ Optional fields with heuristic extraction
    geo_coordinates:        Optional[str]  # from EXIF data
    timestamp_extracted:    Optional[datetime]  # from EXIF
    page_count:            Optional[int]

# ──────────────── EMAIL / MSG ──────────
class EmailDoc(BaseDoc):
    doc_class: Literal["EMAIL_CORRESPONDENCE"]
    sent_date:     datetime
    subject:       str
    sender:        str
    recipients:    List[str]
    body_excerpt:  str

# ──────────────── POLICY / OTHER ───────
class PolicyDoc(BaseDoc):
    doc_class: Literal["POLICY"]
    policy_name:   str
    version:       Optional[str]
    effective_date:Optional[date]
    excerpt:       str

class OtherDoc(BaseDoc):
    doc_class: Literal["OTHER"]
    note: Optional[str]
# ───────── COURT ORDER / JUDGMENT ─────────
class CourtDoc(BaseDoc):
    doc_class: Literal["COURT_ORDER"]
    court_name:     str
    case_number:    str
    filing_date:    date
    judgment_amount:condecimal(max_digits=10, decimal_places=2)
    awarded_to:     Literal["LANDLORD","TENANT","OTHER"]
    possession_date:Optional[date]

# ────── MAINTENANCE ESTIMATE ─────
class EstimateLine(BaseModel):
    work_item: str
    est_amount: condecimal(max_digits=10, decimal_places=2)
class MaintenanceEstimate(BaseDoc):
    doc_class: Literal["MAINTENANCE_ESTIMATE"]
    estimate_id:  str
    vendor_name:  str
    estimate_date:date
    valid_until:  Optional[date]
    line_items:   List[EstimateLine]
    estimated_total: condecimal(max_digits=10, decimal_places=2)

# ────── MOVE-OUT INSPECTION ─────
class RoomScore(BaseModel):
    room: str
    score: int  # 1-5
class MoveOutInspection(BaseDoc):
    doc_class: Literal["MOVE_OUT_INSPECTION"]
    inspection_date: date
    inspector_name:  Optional[str]
    scores:          List[RoomScore]
    photo_links:     Optional[List[str]]

# ────── AUTH FORM ─────
class AuthForm(BaseDoc):
    doc_class: Literal["AUTH_FORM"]
    auth_type: Literal["ACH","DD"]
    account_last4: str
    routing_last4: Optional[str]
    signed_date:   date
    # ★ Optional fields with heuristic extraction
    bank_name:     Optional[str]  # from routing number lookup
    page_count:    Optional[int]

# ────── DEPOSIT REPLACEMENT ADDENDUM (extend LeaseInfo) ─────
# Note: Adding deposit replacement fields to existing LeaseInfo class
# This extends the base LeaseInfo with waiver/insurance fields


# ──────────────── UNION ────────────────
DocUnion = (
    LeaseDoc | LedgerDoc | SDIDoc | InvoiceDoc |
    CollectionLetter | ApplicationDoc |
    PhotoReportDoc | EmailDoc | PolicyDoc | OtherDoc |
    CourtDoc | MaintenanceEstimate | MoveOutInspection | AuthForm
)


