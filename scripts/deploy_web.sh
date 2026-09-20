#!/bin/bash
# Publish the frozen demo (web/, built by scripts/export_static.py) to Vercel as a plain static site.
# There is no server and no secret in it: every solid was built at export time.
#   vercel login                 # once, interactive
#   ./scripts/deploy_web.sh      # PROJECT=<name> to publish under a different subdomain
set -euo pipefail
PROJECT="${PROJECT:-cadabra-cad}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/opt/homebrew/bin:$PATH"

[ -f "$ROOT/web/index.html" ] || { echo "no web/ build: run scripts/export_static.py first" >&2; exit 1; }
vercel whoami >/dev/null 2>&1 || { echo "not logged in: run 'vercel login' first" >&2; exit 1; }

cd "$ROOT/web"
vercel link --yes --project "$PROJECT" >/dev/null
vercel deploy --prod --yes --archive=tgz
echo
echo "live at https://$PROJECT.vercel.app"
