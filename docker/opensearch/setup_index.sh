#!/bin/bash
# Apply index template to a running OpenSearch instance.
# Usage: ./setup_index.sh [opensearch_url]
# Default URL: http://localhost:9200

set -euo pipefail
OPENSEARCH_URL="${1:-http://localhost:9200}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Waiting for OpenSearch at $OPENSEARCH_URL ..."
until curl -sf "$OPENSEARCH_URL/_cluster/health" | grep -qE '"status":"(green|yellow)"'; do
    sleep 5
done

echo "Applying index template ..."
curl -s -X PUT "$OPENSEARCH_URL/_index_template/openalex_works_template" \
    -H "Content-Type: application/json" \
    -d @"$SCRIPT_DIR/index_template.json"
echo ""

echo "Creating index if it does not exist ..."
curl -s -X PUT "$OPENSEARCH_URL/openalex_works" \
    -H "Content-Type: application/json" || true
echo ""

echo "Cluster health:"
curl -s "$OPENSEARCH_URL/_cluster/health?pretty"
