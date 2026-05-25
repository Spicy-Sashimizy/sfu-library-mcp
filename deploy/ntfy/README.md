# Self-hosted ntfy on TrueNAS — off-network alerts via your existing Cloudflare → NPM

A tiny (256 MB / 0.5 CPU) ntfy server for private push notifications under your
own domain. Locked down (`auth-default-access: deny-all` + token) so it's safe
to expose. Reuses your **existing Cloudflare → NPM** path — no new tunnel.

> Recon (2026-05-25): no `cloudflared` runs on the NAS, but NPM does, so your
> Cloudflare is fronting NPM via DNS. These steps add ntfy to that same chain.
> If you actually expose things via a **Cloudflare Tunnel elsewhere**, use the
> "Tunnel route" note in step 4 instead.

## 1. Deploy the container (GATED — run on the NAS; docker needs sudo)
```bash
sudo mkdir -p /mnt/MAIN/sfu-library-mcp/ntfy
# put your domain in config.env (gitignored):
printf 'NTFY_BASE_URL=https://ntfy.YOURDOMAIN\nNTFY_PORT=30280\n' | sudo tee /mnt/MAIN/sfu-library-mcp/ntfy-deploy/config.env
# copy this folder to the NAS then:
cd <this folder on NAS> && sudo docker compose --env-file config.env up -d
sudo docker logs sfu-ntfy --tail 20         # expect "Listening on :80"
```

## 2. Create a token-scoped publisher (what makes exposure safe)
```bash
# a user the monitor + your phone authenticate as, limited to the alert topics:
sudo docker exec -it sfu-ntfy ntfy user add monitor          # set a password
sudo docker exec sfu-ntfy ntfy access monitor 'sfu-truenas-*' rw
sudo docker exec sfu-ntfy ntfy token add monitor             # -> prints tk_xxptoken
```
Keep that `tk_...` token. It goes in the monitor secrets and the phone app.

## 3. Point the monitor at it
```bash
echo -n "https://ntfy.YOURDOMAIN/sfu-truenas-7292faa659" | sudo tee /mnt/MAIN/sfu-library-training/secrets/notify_webhook
echo -n "tk_xxptoken"                                       | sudo tee /mnt/MAIN/sfu-library-training/secrets/notify_token
```
(The monitor's `notify()` sends `Authorization: Bearer <token>` when `notify_token` exists.)

## 4. Expose via your existing Cloudflare → NPM
**Cloudflare DNS:** add a record `ntfy` (CNAME to your existing proxied host, or
A to your public IP), **proxied (orange cloud)** — same as your other subdomains.

**NPM → New Proxy Host:**
- Domain: `ntfy.YOURDOMAIN`
- Forward: `http`  →  `192.168.1.142`  :  `30280`
- **Enable "Websockets Support"** (ntfy's app uses a persistent WS connection — required for instant push).
- SSL tab: request a Let's Encrypt cert (or use your Cloudflare origin cert), Force SSL.

> **Tunnel route (only if you front things with a Cloudflare Tunnel instead):**
> add a Public Hostname `ntfy.YOURDOMAIN` → service `http://192.168.1.142:30280`,
> and enable WebSockets in the tunnel's settings. Everything else is identical.

Cloudflare supports WebSockets through the proxy, so instant push works off-network.

## 5. Phone app
ntfy app → Settings → add server `https://ntfy.YOURDOMAIN`, log in as `monitor`
(or paste the token), then subscribe to topic `sfu-truenas-7292faa659`.

## Test
```bash
curl -H "Authorization: Bearer tk_xxptoken" -H "Title: test" \
     -d "hello from the NAS" https://ntfy.YOURDOMAIN/sfu-truenas-7292faa659
```
Should buzz your phone from anywhere.

## Don't-impede notes
256 MB / 0.5 CPU cap, own dataset dir, single host port in your free range. Does
not touch the 18 production containers or NPM's existing hosts (it only *adds* one).
