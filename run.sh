#!/bin/bash
# Simple script to run the MCP client
cd "$(dirname "$0")"
uv run python src/main.py
