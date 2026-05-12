#!/usr/bin/env bash
# ============================================================
# Set GitHub repository topics for LesionIQ
# ============================================================
#
# Usage:
#   export GITHUB_TOKEN=ghp_your_token_here
#   bash scripts/set_github_topics.sh
#
# Requires a GitHub PAT with "repo" scope.
# Topics can also be set manually at:
#   https://github.com/Astroshreyas1/LesionIQ → About → Topics
# ============================================================

set -euo pipefail

REPO="Astroshreyas1/LesionIQ"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Error: GITHUB_TOKEN is not set."
  echo ""
  echo "Set it with:"
  echo "  export GITHUB_TOKEN=ghp_your_personal_access_token"
  echo ""
  echo "Or set topics manually at:"
  echo "  https://github.com/${REPO} → About (gear icon) → Topics"
  echo ""
  echo "Recommended topics:"
  echo "  deep-learning, medical-imaging, dermoscopy, skin-cancer,"
  echo "  skin-lesion-classification, pytorch, efficientnet, swin-transformer,"
  echo "  explainability, grad-cam, stylegan2, isic, clinical-ai,"
  echo "  dermatology, synthetic-data"
  exit 1
fi

curl -s -X PUT \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${REPO}/topics" \
  -d '{
    "names": [
      "deep-learning",
      "medical-imaging",
      "dermoscopy",
      "skin-cancer",
      "skin-lesion-classification",
      "pytorch",
      "efficientnet",
      "swin-transformer",
      "explainability",
      "grad-cam",
      "stylegan2",
      "isic",
      "clinical-ai",
      "dermatology",
      "synthetic-data"
    ]
  }' | python3 -m json.tool

echo ""
echo "Topics set successfully. View at https://github.com/${REPO}"
