"""DigitalOcean API v2 wrapper — boot-from-snapshot / attach-volume / destroy.

Dry-run by default (cfg.live=False): every mutating call is LOGGED, not sent, and
returns a plausible fake so the whole broker can be exercised end-to-end without
spending a cent of the $200 credit. Flip DEMO_BROKER_LIVE=1 only after the
snapshot + volume IDs are set and the firewall locks :8080 to the NAS IP.

stdlib-only (urllib) so the NAS deploy needs no pip installs.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger("demo_broker.do")
_API = "https://api.digitalocean.com/v2"


class DOError(RuntimeError):
    pass


class DOClient:
    def __init__(self, cfg):
        self.cfg = cfg

    # --- HTTP ---
    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{_API}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.cfg.do_token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise DOError(f"DO {method} {path} -> {e.code}: {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise DOError(f"DO {method} {path} unreachable: {e}") from e

    # --- lifecycle ---
    def create_droplet(self) -> str:
        """Create the demo droplet from the pre-baked snapshot. Returns droplet id."""
        body = {
            "name": self.cfg.do_droplet_name,
            "region": self.cfg.do_region,
            "size": self.cfg.do_size,
            "image": self.cfg.do_snapshot_id,
            "volumes": [self.cfg.do_volume_id] if self.cfg.do_volume_id else [],
            "tags": ["sfu-demo"],
        }
        if not self.cfg.live:
            log.warning("[DRY-RUN] create_droplet %s", json.dumps(body))
            return "dry-run-droplet"
        res = self._req("POST", "/droplets", body)
        did = str(res["droplet"]["id"])
        log.info("created droplet %s (%s, %s)", did, self.cfg.do_size, self.cfg.do_region)
        if self.cfg.do_firewall_id:
            self._req("POST", f"/firewalls/{self.cfg.do_firewall_id}/droplets",
                      {"droplet_ids": [int(did)]})
        return did

    def destroy_droplet(self, droplet_id: str) -> None:
        if not droplet_id:
            return
        if not self.cfg.live:
            log.warning("[DRY-RUN] destroy_droplet %s (volume kept)", droplet_id)
            return
        # Detaching is implicit on destroy; the volume persists (not deleted).
        self._req("DELETE", f"/droplets/{droplet_id}")
        log.info("destroyed droplet %s (volume %s kept)", droplet_id, self.cfg.do_volume_id)

    def droplet_ip(self, droplet_id: str) -> str | None:
        if not self.cfg.live:
            return "127.0.0.1"
        res = self._req("GET", f"/droplets/{droplet_id}")
        for net in res.get("droplet", {}).get("networks", {}).get("v4", []):
            if net.get("type") == "public":
                return net.get("ip_address")
        return None

    def droplet_active(self, droplet_id: str) -> bool:
        """DO-side 'active' status (not the same as app /health-ready)."""
        if not self.cfg.live:
            return True
        res = self._req("GET", f"/droplets/{droplet_id}")
        return res.get("droplet", {}).get("status") == "active"
