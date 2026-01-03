#!/bin/bash
set -e

# Configuration
DOCS_DIR="$(dirname "$0")/../../docs"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "=== Pushing Updates to Overleaf ==="
cd "$DOCS_DIR"

# Check for changes
if git diff-index --quiet HEAD --; then
    # No changes in tracked files, but check if there are untracked files
    if [ -z "$(git status --porcelain)" ]; then
        echo "No changes found to push."
        exit 0
    fi
fi

# Add all changes
echo "--> Staging all changes..."
git add .

# Commit with timestamp
echo "--> Committing..."
# Check if a message was provided as an argument
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
else
    COMMIT_MSG="Update from local environment - $TIMESTAMP"
fi
git commit -m "$COMMIT_MSG"

# Push
echo "--> Pushing to Overleaf..."
git push origin master

echo ""
echo "=== Push Complete! ==="
echo "Your local changes are now live on Overleaf."
