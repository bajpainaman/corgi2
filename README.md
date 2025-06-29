# Document Classifier API v2 - Production Ready

A production-ready FastAPI service that classifies documents using Mistral OCR with comprehensive schema supporting **16 document types** and **80+ structured data points**.

## 🚀 Key Features

### Strip PDF to the Studs
- **Captures every single glyph** from documents
- Page-level raw text extraction with `page_text` array
- Full concatenated text for easy searching with `full_text`
- Zero re-OCR cost - text extracted once, usable forever

### Enhanced Data Extraction
- **Optional fields** with heuristic extraction (NSF fees, forwarding addresses, EXIF data)
- **Raw text persistence** for greppable archives (configurable)
- **Production error handling** with detailed error responses

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export MISTRAL_API_KEY="your_api_key_here"
export SAVE_RAW_TXT="true"  # Optional: Enable raw text archiving
export MAX_FILE_SIZE_MB="50"  # Optional: Max file size (default 50MB)

# Start the API server
python document_classifier_api_v2.py
```

Server runs at `http://localhost:8000`

### Production Environment Variables
- `MISTRAL_API_KEY` - **Required**: Your Mistral API key
- `SAVE_RAW_TXT` - Optional: Enable raw text persistence (`true`/`false`, default: `false`)
- `MAX_FILE_SIZE_MB` - Optional: Maximum file size in MB (default: `50`)
- `MAX_PAGES_PER_CHUNK` - Optional: Pages per chunk for large documents (default: `4`)

## 📊 Document Types Supported

### Core Financial Documents
- **LEASE/ADDENDUM** - Rental agreements with enhanced pet fees, renewal options, deposit waivers
- **LEDGER** - Account statements with aging buckets, payment history
- **SDI/MOVE_OUT_STATEMENT** - Security deposit itemizations with statutory compliance
- **INVOICE** - Vendor bills with PO tracking, payment reconciliation

### Legal & Compliance Documents  
- **COLLECTION_LETTER** - FDCPA-compliant debt collection notices
- **COURT_ORDER** - Court judgments, case numbers, possession dates ⭐ NEW
- **APPLICATION** - Tenant applications with credit, employment, rental history

### Evidence & Documentation
- **PHOTO_REPORT** - Inspection photos with before/after categorization
- **EMAIL_CORRESPONDENCE** - Email communications with sender/recipient tracking
- **MOVE_OUT_INSPECTION** - Property condition scores & photo evidence ⭐ NEW

### Operational Documents
- **MAINTENANCE_ESTIMATE** - Repair estimates with cost breakdowns ⭐ NEW
- **AUTH_FORM** - Banking authorization for deposit returns ⭐ NEW
- **POLICY** - Property management policies and procedures
- **OTHER** - Fallback for unclassified documents

## 🎯 Business Value

### Legal Compliance
- Court judgments that override ledger amounts ✅
- Statutory deadline tracking for deposit returns ✅
- FDCPA compliance validation for collection letters ✅

### Financial Intelligence
- Aging bucket analysis for debt collection prioritization ✅
- Pre-invoice repair estimates for damage justification ✅
- Payment authorization compliance & audit trails ✅

### Evidence Correlation
- Photo evidence linked to damage charges ✅
- Email proof of legal notices and communications ✅
- Room-by-room inspection scores with photo documentation ✅

## 📡 API Endpoints

### Classification
- `POST /classify` - Single document classification
- `POST /classify/batch` - Batch processing (up to 10 files)

### Validation & Testing
- `GET /validate/schema/test` - Run comprehensive schema validation
- `POST /validate/document` - Validate document data against schema
- `POST /validate/extraction` - Extract + validate in one call

### Coverage Analysis
- `GET /coverage/fields/{doc_type}` - Field coverage for specific type
- `GET /coverage/all` - Comprehensive field coverage analysis
- `GET /test/capabilities` - Production readiness testing

### Schema Info
- `GET /schema` - Complete DocUnion schema definition
- `GET /health` - Health check with schema version
- `GET /docs` - Interactive Swagger UI documentation

## 🔧 Production Features

- **16 Document Classes** with comprehensive field extraction
- **80+ Structured Data Points** for maximum legal/financial coverage
- **Strip PDF to the Studs** - Capture every single glyph from documents
- **Optional Fields** - Heuristic extraction (NSF fees, forwarding addresses, EXIF data)
- **Large Document Handling** - Automatic chunking for 8+ page documents
- **Raw Text Persistence** - Greppable archives for full-text search
- **Real-time Validation** - Schema compliance verification
- **Production Error Handling** - Robust error responses and logging
- **Enterprise Testing** - Comprehensive validation & monitoring endpoints
- **Legal Compliance** - FDCPA, statutory deadlines, evidence correlation

## 📁 Project Structure

```
/corgi/
├── document_classifier_api_v2.py  # Main production API
├── main_base_class.py             # Comprehensive schema definitions
├── requirements.txt               # Python dependencies
├── README_API_v2.md              # Detailed API documentation
├── data.csv                      # Sample/test data
└── old_docs/                     # Document samples for testing
```

## 🚀 Deployment Ready

The Document Classifier API v2 is enterprise-ready for processing large document collections with:
- **Legal compliance** tracking (court orders, statutory deadlines)
- **Financial intelligence** (aging analysis, payment reconciliation)  
- **Evidence correlation** (photos, emails, invoices, inspections)
- **Production monitoring** (health checks, validation testing)

Perfect for processing complex property management document collections with comprehensive data extraction and litigation-ready intelligence.# corgi2
