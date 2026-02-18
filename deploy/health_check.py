"""Production health check for Vanse Data Stack on TrueNAS CE.

Checks:
- Docker containers running
- Database connectivity and migrations
- Disk space
- Recent backups
- Recent scraper activity

Usage:
    python -m deploy.health_check              # Local mode (run on TrueNAS)
    python -m deploy.health_check --remote     # Remote mode (SSH from dev container)
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import click


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


EXPECTED_CONTAINERS = ["vanse_db", "vanse_scrapers", "vanse_n8n", "vanse_metabase"]

COMPOSE_FILE = "docker-compose.prod.yml"
BACKUP_DIR = "/mnt/tank/vanse/backups"
DISK_WARN_PERCENT = 80
DISK_FAIL_PERCENT = 95
BACKUP_WARN_HOURS = 48
SCRAPER_WARN_HOURS = 48


def _run_cmd(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run a command and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


def _run_remote_cmd(cmd: str, **kwargs: str) -> tuple[int, str]:
    """Run a command on the remote TrueNAS host via SSH."""
    ssh_key = kwargs.get("ssh_key", "")
    host = kwargs.get("host", "")
    user = kwargs.get("user", "")
    ssh_cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        f"{user}@{host}",
        cmd,
    ]
    return _run_cmd(ssh_cmd, timeout=30)


def check_containers_running(remote: bool = False, **ssh_kwargs: str) -> HealthCheckResult:
    """Check that all expected Docker containers are running."""
    cmd = ["docker", "compose", "-f", COMPOSE_FILE, "ps", "--format", "{{.Name}} {{.Status}}"]

    if remote:
        app_path = ssh_kwargs.get("app_path", "/mnt/tank/vanse/app")
        shell_cmd = f"cd {app_path} && docker compose -f {COMPOSE_FILE} ps --format '{{{{.Name}}}} {{{{.Status}}}}'"
        rc, output = _run_remote_cmd(
            shell_cmd, **{k: v for k, v in ssh_kwargs.items() if k != "app_path"}
        )
    else:
        rc, output = _run_cmd(cmd)

    if rc != 0:
        return HealthCheckResult(
            name="Containers",
            status=HealthStatus.FAIL,
            message=f"Failed to check containers: {output}",
        )

    running: list[str] = []
    not_running: list[str] = []
    details: list[str] = []

    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        name = parts[0] if parts else ""
        status_text = parts[1] if len(parts) > 1 else ""
        details.append(f"{name}: {status_text}")

        if name in EXPECTED_CONTAINERS:
            if "up" in status_text.lower():
                running.append(name)
            else:
                not_running.append(name)

    missing = [c for c in EXPECTED_CONTAINERS if c not in running and c not in not_running]
    not_running.extend(missing)

    if not_running:
        return HealthCheckResult(
            name="Containers",
            status=HealthStatus.FAIL,
            message=f"Containers not running: {', '.join(not_running)}",
            details=details,
        )

    return HealthCheckResult(
        name="Containers",
        status=HealthStatus.OK,
        message=f"All {len(running)} containers running",
        details=details,
    )


def check_db_connectivity(remote: bool = False, **ssh_kwargs: str) -> HealthCheckResult:
    """Check PostgreSQL is reachable and responding."""
    cmd = [
        "docker",
        "exec",
        "vanse_db",
        "pg_isready",
        "-U",
        "vanse",
        "-d",
        "vanse_leads",
    ]

    if remote:
        shell_cmd = "docker exec vanse_db pg_isready -U vanse -d vanse_leads"
        rc, output = _run_remote_cmd(
            shell_cmd, **{k: v for k, v in ssh_kwargs.items() if k != "app_path"}
        )
    else:
        rc, output = _run_cmd(cmd)

    if rc != 0:
        return HealthCheckResult(
            name="Database",
            status=HealthStatus.FAIL,
            message=f"Database not reachable: {output}",
        )

    return HealthCheckResult(
        name="Database",
        status=HealthStatus.OK,
        message="Database accepting connections",
    )


def check_migrations_applied(remote: bool = False, **ssh_kwargs: str) -> HealthCheckResult:
    """Check that database migrations have been applied."""
    sql = "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'companies';"
    cmd = [
        "docker",
        "exec",
        "vanse_db",
        "psql",
        "-U",
        "vanse",
        "-d",
        "vanse_leads",
        "-t",
        "-c",
        sql,
    ]

    if remote:
        shell_cmd = f'docker exec vanse_db psql -U vanse -d vanse_leads -t -c "{sql}"'
        rc, output = _run_remote_cmd(
            shell_cmd, **{k: v for k, v in ssh_kwargs.items() if k != "app_path"}
        )
    else:
        rc, output = _run_cmd(cmd)

    if rc != 0:
        return HealthCheckResult(
            name="Migrations",
            status=HealthStatus.FAIL,
            message=f"Cannot check migrations: {output}",
        )

    count = output.strip()
    if count == "1":
        return HealthCheckResult(
            name="Migrations",
            status=HealthStatus.OK,
            message="Core tables exist (companies table found)",
        )

    return HealthCheckResult(
        name="Migrations",
        status=HealthStatus.FAIL,
        message="Core tables missing — run migrations",
    )


def check_disk_space(path: str = "/mnt/tank/vanse") -> HealthCheckResult:
    """Check disk usage at the data path."""
    try:
        stat = os.statvfs(path)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        used = total - free
        usage_pct = (used / total * 100) if total > 0 else 0

        total_gb = total / (1024**3)
        free_gb = free / (1024**3)
        details = [f"Total: {total_gb:.1f} GB, Free: {free_gb:.1f} GB, Used: {usage_pct:.1f}%"]

        if usage_pct >= DISK_FAIL_PERCENT:
            return HealthCheckResult(
                name="Disk Space",
                status=HealthStatus.FAIL,
                message=f"Disk usage critical: {usage_pct:.1f}%",
                details=details,
            )
        if usage_pct >= DISK_WARN_PERCENT:
            return HealthCheckResult(
                name="Disk Space",
                status=HealthStatus.WARN,
                message=f"Disk usage high: {usage_pct:.1f}%",
                details=details,
            )

        return HealthCheckResult(
            name="Disk Space",
            status=HealthStatus.OK,
            message=f"Disk usage: {usage_pct:.1f}%",
            details=details,
        )
    except OSError as e:
        return HealthCheckResult(
            name="Disk Space",
            status=HealthStatus.WARN,
            message=f"Cannot check disk space: {e}",
        )


def check_recent_backup(backup_dir: str = BACKUP_DIR) -> HealthCheckResult:
    """Check that a backup exists within the last BACKUP_WARN_HOURS."""
    try:
        if not os.path.isdir(backup_dir):
            return HealthCheckResult(
                name="Backups",
                status=HealthStatus.WARN,
                message=f"Backup directory not found: {backup_dir}",
            )

        backups = sorted(
            [f for f in os.listdir(backup_dir) if f.startswith("db_") and f.endswith(".sql.gz")],
            reverse=True,
        )

        if not backups:
            return HealthCheckResult(
                name="Backups",
                status=HealthStatus.FAIL,
                message="No database backups found",
            )

        latest = backups[0]
        latest_path = os.path.join(backup_dir, latest)
        mtime = datetime.fromtimestamp(os.path.getmtime(latest_path))
        age = datetime.now() - mtime
        age_hours = age.total_seconds() / 3600

        details = [f"Latest: {latest} ({age_hours:.1f} hours ago)"]

        if age_hours > BACKUP_WARN_HOURS:
            return HealthCheckResult(
                name="Backups",
                status=HealthStatus.WARN,
                message=f"Latest backup is {age_hours:.0f} hours old (threshold: {BACKUP_WARN_HOURS}h)",
                details=details,
            )

        return HealthCheckResult(
            name="Backups",
            status=HealthStatus.OK,
            message=f"Latest backup: {age_hours:.1f} hours ago",
            details=details,
        )
    except OSError as e:
        return HealthCheckResult(
            name="Backups",
            status=HealthStatus.WARN,
            message=f"Cannot check backups: {e}",
        )


def check_scraper_activity(remote: bool = False, **ssh_kwargs: str) -> HealthCheckResult:
    """Check if any scraper has produced data recently."""
    sql = (
        "SELECT source, MAX(created_at) as latest "
        "FROM companies GROUP BY source ORDER BY latest DESC LIMIT 5;"
    )
    cmd = [
        "docker",
        "exec",
        "vanse_db",
        "psql",
        "-U",
        "vanse",
        "-d",
        "vanse_leads",
        "-t",
        "-c",
        sql,
    ]

    if remote:
        shell_cmd = f'docker exec vanse_db psql -U vanse -d vanse_leads -t -c "{sql}"'
        rc, output = _run_remote_cmd(
            shell_cmd, **{k: v for k, v in ssh_kwargs.items() if k != "app_path"}
        )
    else:
        rc, output = _run_cmd(cmd)

    if rc != 0:
        return HealthCheckResult(
            name="Scraper Activity",
            status=HealthStatus.WARN,
            message=f"Cannot check scraper activity: {output}",
        )

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return HealthCheckResult(
            name="Scraper Activity",
            status=HealthStatus.WARN,
            message="No company records found — scrapers may not have run yet",
        )

    details = [f"  {line}" for line in lines]
    return HealthCheckResult(
        name="Scraper Activity",
        status=HealthStatus.OK,
        message=f"Found activity from {len(lines)} sources",
        details=details,
    )


def run_all_checks(
    remote: bool = False,
    disk_path: str = "/mnt/tank/vanse",
    backup_dir: str = BACKUP_DIR,
    **ssh_kwargs: str,
) -> list[HealthCheckResult]:
    """Run all health checks."""
    ssh_args = {k: v for k, v in ssh_kwargs.items() if v}

    results = [
        check_containers_running(remote=remote, **ssh_args),
        check_db_connectivity(remote=remote, **ssh_args),
        check_migrations_applied(remote=remote, **ssh_args),
        check_scraper_activity(remote=remote, **ssh_args),
    ]

    # Disk and backup checks only make sense in local mode
    if not remote:
        results.append(check_disk_space(path=disk_path))
        results.append(check_recent_backup(backup_dir=backup_dir))

    return results


def _status_symbol(status: HealthStatus) -> str:
    if status == HealthStatus.OK:
        return " OK "
    elif status == HealthStatus.WARN:
        return "WARN"
    else:
        return "FAIL"


@click.command()
@click.option("--remote", is_flag=True, help="Run checks via SSH to TrueNAS")
@click.option("--host", default=None, help="TrueNAS hostname/IP (default: TRUENAS_HOST env)")
@click.option("--user", default=None, help="SSH user (default: TRUENAS_USER env)")
@click.option("--ssh-key", default=None, help="SSH key path")
def main(remote: bool, host: str | None, user: str | None, ssh_key: str | None) -> None:
    """Run production health checks for Vanse Data Stack."""
    ssh_kwargs: dict[str, str] = {}
    if remote:
        ssh_kwargs["host"] = host or os.environ.get("TRUENAS_HOST", "192.168.1.100")
        ssh_kwargs["user"] = user or os.environ.get("TRUENAS_USER", "vanse")
        ssh_kwargs["ssh_key"] = ssh_key or os.environ.get(
            "SSH_KEY", "/workspaces/vansedataadstackshit/.ssh/id_ed25519"
        )
        ssh_kwargs["app_path"] = "/mnt/tank/vanse/app"

    results = run_all_checks(remote=remote, **ssh_kwargs)

    mode = "Remote (SSH)" if remote else "Local"
    click.echo(f"\n{'='*60}")
    click.echo(f"  Vanse Data Stack Health Check — {mode}")
    click.echo(f"{'='*60}\n")

    failures = 0
    warnings = 0
    for r in results:
        symbol = _status_symbol(r.status)
        click.echo(f"  [{symbol}] {r.name}: {r.message}")
        for d in r.details:
            click.echo(f"         {d}")
        if r.status == HealthStatus.FAIL:
            failures += 1
        elif r.status == HealthStatus.WARN:
            warnings += 1

    click.echo(f"\n{'='*60}")
    total = len(results)
    ok = sum(1 for r in results if r.status == HealthStatus.OK)
    click.echo(f"  {ok}/{total} ok, {warnings} warnings, {failures} failures")
    click.echo(f"{'='*60}\n")

    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
