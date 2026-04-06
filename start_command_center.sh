#!/bin/bash
echo "🚀 Starting SciML Research Command Center..."
echo "API: http://localhost:8000"
echo "UI:  file://$(pwd)/ui/dashboard.html"
echo ""
uv run python3 app.py
