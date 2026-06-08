#!/usr/bin/env bash
# Idempotently expose the NAS-hosted SFU Library MCP server to Claude's remote
# connectors via the EXISTING Cloudflare tunnel, locked to Anthropic's IPs.
#
# Reads everything from .env (gitignored — no secrets in this file):
#   CLOUDFLARE_API_TOKEN  (scopes: Cloudflare Tunnel:Edit, DNS:Edit, WAF:Edit, Account:Read)
#   CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_ZONE_ID CLOUDFLARE_TUNNEL_ID
#   MCP_PUBLIC_HOSTNAME   MCP_TUNNEL_ORIGIN   ANTHROPIC_ALLOW_CIDR
#
# Safe to re-run: WAF rule create-or-append, tunnel ingress is MERGED (existing
# hostnames preserved), DNS create-or-skip. Connector URL = https://$HOST/mcp
set -euo pipefail
cd "$(dirname "$0")/.."

gv(){ grep -E "^[[:space:]]*$1=" .env | head -1 | sed -E "s/^[[:space:]]*$1=//; s/[[:space:]]*#.*//; s/[[:space:]]*$//"; }
TOK=$(gv CLOUDFLARE_API_TOKEN); ACCT=$(gv CLOUDFLARE_ACCOUNT_ID); ZONE=$(gv CLOUDFLARE_ZONE_ID)
TUN=$(gv CLOUDFLARE_TUNNEL_ID); HOST=$(gv MCP_PUBLIC_HOSTNAME); ORIGIN=$(gv MCP_TUNNEL_ORIGIN); CIDR=$(gv ANTHROPIC_ALLOW_CIDR)
AUTH=(-H "Authorization: Bearer $TOK" -H "Content-Type: application/json")
EXPR="(http.host eq \"$HOST\" and not ip.src in {$CIDR})"
api="https://api.cloudflare.com/client/v4"
ok(){ [ "$(echo "$1"|jq -r .success)" = "true" ]; }

echo "### 1. WAF allowlist (block non-Anthropic on $HOST) ###"
EP=$(curl -s "${AUTH[@]}" "$api/zones/$ZONE/rulesets/phases/http_request_firewall_custom/entrypoint")
if ok "$EP"; then
  if [ "$(echo "$EP"|jq -r --arg e "$EXPR" '[.result.rules[]?|select(.expression==$e)]|length')" -ge 1 ]; then
    echo "  rule already present"; R="$EP"
  else
    R=$(curl -s -X POST "${AUTH[@]}" "$api/zones/$ZONE/rulesets/phases/http_request_firewall_custom/entrypoint/rules" \
        -d "$(jq -n --arg e "$EXPR" '{action:"block",description:"MCP: allow only Anthropic connector IPs",expression:$e}')")
  fi
else
  R=$(curl -s -X PUT "${AUTH[@]}" "$api/zones/$ZONE/rulesets/phases/http_request_firewall_custom/entrypoint" \
      -d "$(jq -n --arg e "$EXPR" '{rules:[{action:"block",description:"MCP: allow only Anthropic connector IPs",expression:$e}]}')")
fi
ok "$R" || { echo "  WAF FAILED: $(echo "$R"|jq -c .errors)"; exit 1; }
echo "  ok"

echo "### 2. Tunnel ingress (merge $HOST -> $ORIGIN, preserve others) ###"
CFG=$(curl -s "${AUTH[@]}" "$api/accounts/$ACCT/cfd_tunnel/$TUN/configurations" | jq '.result.config')
NEWCFG=$(echo "$CFG" | jq --arg h "$HOST" --arg svc "$ORIGIN" \
  '.ingress = ((.ingress|map(select(.hostname!=$h))) | (.[:-1] + [{"hostname":$h,"service":$svc}] + [.[-1]]))')
PUTR=$(curl -s -X PUT "${AUTH[@]}" "$api/accounts/$ACCT/cfd_tunnel/$TUN/configurations" -d "$(jq -n --argjson c "$NEWCFG" '{config:$c}')")
ok "$PUTR" || { echo "  ingress FAILED: $(echo "$PUTR"|jq -c .errors)"; exit 1; }
echo "$PUTR" | jq -r '.result.config.ingress[] | "    \(.hostname // "[catch-all]") -> \(.service)"'

echo "### 3. DNS CNAME ###"
EX=$(curl -s "${AUTH[@]}" "$api/zones/$ZONE/dns_records?name=$HOST")
if [ "$(echo "$EX"|jq -r '.result|length')" -ge 1 ]; then echo "  exists"; else
  DR=$(curl -s -X POST "${AUTH[@]}" "$api/zones/$ZONE/dns_records" \
       -d "$(jq -n --arg n "$HOST" --arg c "$TUN.cfargotunnel.com" '{type:"CNAME",name:$n,content:$c,proxied:true,comment:"MCP remote connector"}')")
  ok "$DR" || { echo "  DNS FAILED: $(echo "$DR"|jq -c .errors)"; exit 1; }; echo "  created"
fi

echo "### 4. Edge probe (403 from a non-Anthropic IP = allowlist working) ###"
EDGE=$(curl -s "https://1.1.1.1/dns-query?name=$HOST&type=A" -H 'accept: application/dns-json' | jq -r '.Answer//[]|map(select(.type==1))|.[0].data // empty')
[ -n "$EDGE" ] && curl -s --resolve "$HOST:443:$EDGE" -o /dev/null -w "  https://$HOST/health -> HTTP %{http_code} (403 expected here)\n" "https://$HOST/health"
echo "Connector URL: https://$HOST/mcp"
