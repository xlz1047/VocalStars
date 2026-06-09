#!/bin/bash
# Quick start script for VocalStars development

set -e

echo "🎤 VocalStars Development Setup"
echo "================================"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

echo "✅ Node.js $(node --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python $(python3 --version)"

# Setup Backend
echo ""
echo "Setting up Backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate || . venv/Scripts/activate
    pip install -r requirements.txt
    if [ -f "requirements-ml.txt" ]; then
        pip install -r requirements-ml.txt
    fi
else
    source venv/bin/activate || . venv/Scripts/activate
fi

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp ../.env.example .env
    echo "⚠️  Backend .env created - configure database connection"
fi

cd ..
echo "✅ Backend setup complete"

# Setup Frontend (new_frontend)
echo ""
echo "Setting up New Frontend..."
cd new_frontend

# Install dependencies
npm install

# Copy .env.local if not exists
if [ ! -f ".env.local" ]; then
    cp .env.local.example .env.local
    echo "⚠️  Frontend .env.local created with defaults"
fi

cd ..
echo "✅ Frontend setup complete"

echo ""
echo "================================"
echo "🚀 Ready to start development!"
echo ""
echo "To start development servers:"
echo ""
echo "  Backend (Terminal 1):"
echo "    cd backend"
echo "    source venv/bin/activate"
echo "    python -m uvicorn app.main:app --reload"
echo ""
echo "  Frontend (Terminal 2):"
echo "    cd new_frontend"
echo "    npm run dev"
echo ""
echo "  Frontend will be available at: http://localhost:3000"
echo "  Backend API at: http://localhost:8000"
echo "  API docs at: http://localhost:8000/docs"
echo ""
