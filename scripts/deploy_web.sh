#!/bin/bash
# Build the published site (site/, Next.js static export) and put it on Vercel.
# There is no server and no secret in it: scripts/export_static.py built every solid ahead of time.
#   uv run python scripts/export_static.py    # refresh site/public/data from a running demo, if the data changed
#   vercel login                              # once, interactive
#   ./scripts/deploy_web.sh                   # PROJECT=<name> to publish under a different subdomain
set -euo pipefail
PROJECT="${PROJECT:-cadabra-cad}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/opt/homebrew/bin:$PATH"
unset -f npm node 2>/dev/null || true   # the shell's nvm wrappers recurse under a non-interactive shell

[ -d "$ROOT/site/public/data" ] || { echo "no exported data: run scripts/export_static.py first" >&2; exit 1; }
vercel whoami >/dev/null 2>&1 || { echo "not logged in: run 'vercel login' first" >&2; exit 1; }

cd "$ROOT/site"
[ -d node_modules ] || npm install --no-audit --no-fund
./node_modules/.bin/next build

# Deployed prebuilt, as plain files. cleanUrls gives /how-it-works rather than /how-it-works.html.
cat > out/vercel.json <<'JSON'
{
  "cleanUrls": true,
  "headers": [
    { "source": "/data/mesh/(.*)", "headers": [{ "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }] },
    { "source": "/_next/static/(.*)", "headers": [{ "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }] }
  ]
}
JSON

cd out
vercel link --yes --project "$PROJECT" >/dev/null
vercel deploy --prod --yes --archive=tgz
echo
echo "live at https://$PROJECT.vercel.app"
