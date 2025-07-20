#!/bin/bash
# Setup script for MCP Platform

set -e

echo "🚀 Setting up MCP Platform..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please create one with your API keys:"
    echo "   echo 'LLM_API_KEY=your_key_here' > .env"
    echo "   Continuing with setup..."
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --extra dev

# Make run script executable
chmod +x run.sh

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Run: ./run.sh"
