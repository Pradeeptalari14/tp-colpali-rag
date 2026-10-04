#!/usr/bin/env bash
set -euo pipefail
echo "Validating ColPali pipeline..."
python3 -m py_compile colpali_retriever.py
python3 -c "import json; json.load(open('qdrant_multivector_schema.json'))"
echo "✓ Validation clean."
