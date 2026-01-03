#!/bin/bash
set -e

echo "=== Pulling Updates from Overleaf ==="
# Navigate to the docs directory where the Overleaf git repo lives
cd "$(dirname "$0")/../../docs"

# Pull changes
git pull origin master

echo ""
echo "=== Update Complete! ==="
echo "Your local 'docs/' folder is now in sync with Overleaf."
