#!/bin/bash
# Production startup script for Document Classifier API v2

set -e

echo "🚀 Starting Document Classifier API v2 - Production"
echo "================================================="

# Check required environment variables
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "❌ Error: MISTRAL_API_KEY environment variable is required"
    exit 1
fi

# Set production defaults
export SAVE_RAW_TXT="${SAVE_RAW_TXT:-false}"
export MAX_FILE_SIZE_MB="${MAX_FILE_SIZE_MB:-50}"
export MAX_PAGES_PER_CHUNK="${MAX_PAGES_PER_CHUNK:-4}"

echo "📋 Configuration:"
echo "   API Key: ${MISTRAL_API_KEY:0:8}..."
echo "   Raw Text Archiving: $SAVE_RAW_TXT"
echo "   Max File Size: ${MAX_FILE_SIZE_MB}MB"
echo "   Max Pages Per Chunk: $MAX_PAGES_PER_CHUNK"

# Create raw_ocr directory if archiving is enabled
if [ "$SAVE_RAW_TXT" = "true" ]; then
    mkdir -p raw_ocr
    echo "📁 Raw text archiving enabled: ./raw_ocr/"
fi

echo ""
echo "🌐 Starting server on http://0.0.0.0:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo ""

# Start the server with production settings
python3 document_classifier_api_v2.py