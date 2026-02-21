"""Production health check for SFU Library MCP on TrueNAS.

Checks:
- Docker container running and healthy
- HTTP endpoint responsive
- Memory usage within limits
- Disk space adequate
- Secrets mounted

Usage:
    python deploy/health_check.py              # Remote mode (SSH from dev container)
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum


class HealthStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    details: list[str] = field(default_factory=list)


SSH_ALIAS = "truenas"
CONTAINER_NAME = "sfu-library-mcp"


def _ssh_cmd(cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run a command on TrueNAS via SSH."""
    try:
        result = subprocess.run(
            ["ssh", SSH_ALIAS, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except FileNotFoundError:
        return 1, "SSH not available"


def check_ssh() -> HealthCheckResult:
    """Check SSH connectivity to TrueNAS."""
    rc, output = _ssh_cmd("echo OK")
    if rc != 0 or output != "OK":
        return HealthCheckResult("SSH", HealthStatus.FAIL, f"Cannot reach TrueNAS: {output}")
    return HealthCheckResult("SSH", HealthStatus.OK, "Connected")


def check_container_running() -> HealthCheckResult:
    """Check that the MCP container is running."""
    rc, output = _ssh_cmd(f"sudo docker ps --filter name={CONTAINER_NAME} --format '{{{{.Status}}}}'")
    if rc != 0 or not output:
        return HealthCheckResult("Container", HealthStatus.FAIL, "Container not running")
    return HealthCheckResult("Container", HealthStatus.OK, f"Running: {output}")


def check_container_healthy() -> HealthCheckResult:
    """Check Docker health check status."""
    rc, output = _ssh_cmd(f"sudo docker inspect --format='{{{{.State.Health.Status}}}}' {CONTAINER_NAME}")
    if rc != 0:
        return HealthCheckResult("Health", HealthStatus.FAIL, f"Cannot inspect: {output}")
    if output == "healthy":
        return HealthCheckResult("Health", HealthStatus.OK, "Healthy")
    return HealthCheckResult("Health", HealthStatus.FAIL, f"Status: {output}")


def check_http_endpoint() -> HealthCheckResult:
    """Check that the HTTP health endpoint responds."""
    rc, output = _ssh_cmd("curl -sf http://localhost:8080/health")
    if rc != 0:
        return HealthCheckResult("HTTP", HealthStatus.FAIL, "Health endpoint not responding")
    return HealthCheckResult("HTTP", HealthStatus.OK, f"Response: {output}")


def check_memory() -> HealthCheckResult:
    """Check container memory usage."""
    rc, output = _ssh_cmd(f"sudo docker stats {CONTAINER_NAME} --no-stream --format '{{{{.MemUsage}}}}'")
    if rc != 0:
        return HealthCheckResult("Memory", HealthStatus.WARN, f"Cannot check: {output}")
    return HealthCheckResult("Memory", HealthStatus.OK, f"Usage: {output}")


def check_disk() -> HealthCheckResult:
    """Check available disk space on /mnt/MAIN."""
    rc, output = _ssh_cmd("df -BG /mnt/MAIN | awk 'NR==2{print $4}'")
    if rc != 0:
        return HealthCheckResult("Disk", HealthStatus.WARN, f"Cannot check: {output}")
    try:
        free_gb = int(output.replace("G", ""))
        if free_gb < 5:
            return HealthCheckResult("Disk", HealthStatus.FAIL, f"Only {free_gb}GB free")
        return HealthCheckResult("Disk", HealthStatus.OK, f"{free_gb}GB free")
    except ValueError:
        return HealthCheckResult("Disk", HealthStatus.WARN, f"Unexpected output: {output}")


def check_secrets() -> HealthCheckResult:
    """Check that all 6 secrets are mounted in the container."""
    rc, output = _ssh_cmd(f"sudo docker exec {CONTAINER_NAME} ls /run/secrets/")
    if rc != 0:
        return HealthCheckResult("Secrets", HealthStatus.FAIL, f"Cannot list secrets: {output}")
    files = output.strip().split("\n")
    expected = {"sfu_username", "sfu_password", "sfu_mfa_secret", "sfu_mfa_device_name", "zotero_api_key", "zotero_user_id"}
    found = set(files)
    missing = expected - found
    if missing:
        return HealthCheckResult("Secrets", HealthStatus.FAIL, f"Missing: {missing}", details=files)
    return HealthCheckResult("Secrets", HealthStatus.OK, f"All {len(expected)} secrets mounted")


def run_all_checks() -> list[HealthCheckResult]:
    """Run all health checks."""
    return [
        check_ssh(),
        check_container_running(),
        check_container_healthy(),
        check_http_endpoint(),
        check_memory(),
        check_disk(),
        check_secrets(),
    ]


def _status_symbol(status: HealthStatus) -> str:
    if status == HealthStatus.OK:
        return " OK "
    elif status == HealthStatus.WARN:
        return "WARN"
    else:
        return "FAIL"


def main() -> None:
    """Run all health checks and print results."""
    results = run_all_checks()

    print(f"\n{'='*60}")
    print(f"  SFU Library MCP — Health Check")
    print(f"{'='*60}\n")

    failures = 0
    warnings = 0
    for r in results:
        symbol = _status_symbol(r.status)
        print(f"  [{symbol}] {r.name}: {r.message}")
        for d in r.details:
            print(f"         {d}")
        if r.status == HealthStatus.FAIL:
            failures += 1
        elif r.status == HealthStatus.WARN:
            warnings += 1

    print(f"\n{'='*60}")
    total = len(results)
    ok = sum(1 for r in results if r.status == HealthStatus.OK)
    print(f"  {ok}/{total} ok, {warnings} warnings, {failures} failures")
    print(f"{'='*60}\n")

    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
