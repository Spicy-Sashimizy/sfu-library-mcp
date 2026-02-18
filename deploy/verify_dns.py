"""DNS record verification for Vanse email infrastructure.

Checks SPF, DKIM, DMARC, and MX records for:
- vanse.vancitylandscaper.com (cold outreach subdomain)
- vancitylandscaper.com (root domain — must stay clean)
- vanseindustry.com (business domain — iCloud Custom Email)

Usage:
    python -m deploy.verify_dns
    python -m deploy.verify_dns --domain vanse.vancitylandscaper.com
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import StrEnum

import click
import dns.resolver


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class DnsCheckResult:
    name: str
    status: Status
    message: str
    details: list[str] = field(default_factory=list)


def _resolve(qname: str, rdtype: str, timeout: float = 10.0) -> list[str]:
    """Resolve DNS records, returning list of string values."""
    try:
        answers = dns.resolver.resolve(qname, rdtype, lifetime=timeout)
        return [rdata.to_text() for rdata in answers]
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
        return []
    except dns.exception.Timeout:
        return []


def check_spf(
    domain: str, required_includes: list[str], forbidden_includes: list[str]
) -> DnsCheckResult:
    """Verify SPF record contains required includes and excludes forbidden ones."""
    records = _resolve(domain, "TXT")
    spf_records = [r for r in records if "v=spf1" in r.lower()]

    if not spf_records:
        return DnsCheckResult(
            name=f"SPF ({domain})",
            status=Status.FAIL,
            message="No SPF record found",
        )

    if len(spf_records) > 1:
        return DnsCheckResult(
            name=f"SPF ({domain})",
            status=Status.WARN,
            message=f"Multiple SPF records found ({len(spf_records)}), should have exactly 1",
            details=spf_records,
        )

    spf = spf_records[0].lower()
    details = [f"Record: {spf_records[0]}"]
    missing = [inc for inc in required_includes if inc.lower() not in spf]
    present_forbidden = [inc for inc in forbidden_includes if inc.lower() in spf]

    if present_forbidden:
        return DnsCheckResult(
            name=f"SPF ({domain})",
            status=Status.FAIL,
            message=f"SPF contains forbidden includes: {', '.join(present_forbidden)}",
            details=details,
        )

    if missing:
        return DnsCheckResult(
            name=f"SPF ({domain})",
            status=Status.FAIL,
            message=f"SPF missing required includes: {', '.join(missing)}",
            details=details,
        )

    return DnsCheckResult(
        name=f"SPF ({domain})",
        status=Status.PASS,
        message="SPF record valid",
        details=details,
    )


def check_dmarc(domain: str, expected_policy: str | None = None) -> DnsCheckResult:
    """Verify DMARC record exists and optionally check policy."""
    dmarc_domain = f"_dmarc.{domain}"
    records = _resolve(dmarc_domain, "TXT")
    dmarc_records = [r for r in records if "v=dmarc1" in r.lower()]

    if not dmarc_records:
        return DnsCheckResult(
            name=f"DMARC ({domain})",
            status=Status.FAIL,
            message="No DMARC record found",
        )

    dmarc = dmarc_records[0].lower()
    details = [f"Record: {dmarc_records[0]}"]

    if expected_policy:
        policy_str = f"p={expected_policy}".lower()
        if policy_str not in dmarc:
            return DnsCheckResult(
                name=f"DMARC ({domain})",
                status=Status.WARN,
                message=f"DMARC policy is not '{expected_policy}'",
                details=details,
            )

    return DnsCheckResult(
        name=f"DMARC ({domain})",
        status=Status.PASS,
        message="DMARC record valid",
        details=details,
    )


def check_mx(domain: str, expected_hosts: list[str]) -> DnsCheckResult:
    """Verify MX records point to expected mail servers."""
    records = _resolve(domain, "MX")

    if not records:
        return DnsCheckResult(
            name=f"MX ({domain})",
            status=Status.FAIL,
            message="No MX records found",
        )

    details = [f"MX: {r}" for r in records]
    mx_lower = " ".join(r.lower() for r in records)

    missing = [h for h in expected_hosts if h.lower() not in mx_lower]
    if missing:
        return DnsCheckResult(
            name=f"MX ({domain})",
            status=Status.FAIL,
            message=f"MX missing expected hosts: {', '.join(missing)}",
            details=details,
        )

    return DnsCheckResult(
        name=f"MX ({domain})",
        status=Status.PASS,
        message="MX records valid",
        details=details,
    )


def check_dkim(domain: str, selectors: list[str]) -> DnsCheckResult:
    """Check DKIM records exist for given selectors."""
    found: list[str] = []
    missing: list[str] = []

    for selector in selectors:
        dkim_domain = f"{selector}._domainkey.{domain}"
        # DKIM can be TXT or CNAME
        txt_records = _resolve(dkim_domain, "TXT")
        cname_records = _resolve(dkim_domain, "CNAME")

        if txt_records or cname_records:
            found.append(selector)
        else:
            missing.append(selector)

    if not found and missing:
        return DnsCheckResult(
            name=f"DKIM ({domain})",
            status=Status.FAIL,
            message=f"No DKIM records found for selectors: {', '.join(missing)}",
        )

    if missing:
        return DnsCheckResult(
            name=f"DKIM ({domain})",
            status=Status.WARN,
            message=f"DKIM found for {', '.join(found)}; missing for {', '.join(missing)}",
        )

    return DnsCheckResult(
        name=f"DKIM ({domain})",
        status=Status.PASS,
        message=f"DKIM records found for all selectors: {', '.join(found)}",
    )


def verify_outreach_subdomain() -> list[DnsCheckResult]:
    """Verify vanse.vancitylandscaper.com (cold outreach subdomain)."""
    domain = "vanse.vancitylandscaper.com"
    results = [
        check_spf(
            domain,
            required_includes=["amazonses.com", "mailgun.org"],
            forbidden_includes=[],
        ),
        check_dmarc(domain),
        check_dkim(domain, selectors=["smtp"]),
    ]
    return results


def verify_root_domain() -> list[DnsCheckResult]:
    """Verify vancitylandscaper.com (root — must NOT authorize SES/Mailgun)."""
    domain = "vancitylandscaper.com"
    results = [
        check_spf(
            domain,
            required_includes=[],
            forbidden_includes=["amazonses.com", "mailgun.org"],
        ),
        check_dmarc(domain),
    ]
    return results


def verify_business_domain() -> list[DnsCheckResult]:
    """Verify vanseindustry.com (business domain — iCloud Custom Email)."""
    domain = "vanseindustry.com"
    results = [
        check_spf(
            domain,
            required_includes=["icloud.com"],
            forbidden_includes=["amazonses.com", "mailgun.org"],
        ),
        check_dmarc(domain, expected_policy="reject"),
        check_mx(domain, expected_hosts=["icloud.com"]),
        check_dkim(domain, selectors=["sig1"]),
    ]
    return results


def verify_all() -> list[DnsCheckResult]:
    """Run all DNS verification checks."""
    results: list[DnsCheckResult] = []
    results.extend(verify_outreach_subdomain())
    results.extend(verify_root_domain())
    results.extend(verify_business_domain())
    return results


def _status_symbol(status: Status) -> str:
    if status == Status.PASS:
        return "PASS"
    elif status == Status.WARN:
        return "WARN"
    else:
        return "FAIL"


@click.command()
@click.option(
    "--domain",
    type=click.Choice(
        ["all", "vanse.vancitylandscaper.com", "vancitylandscaper.com", "vanseindustry.com"]
    ),
    default="all",
    help="Domain to verify (default: all)",
)
def main(domain: str) -> None:
    """Verify DNS records for Vanse email infrastructure."""
    if domain == "all":
        results = verify_all()
    elif domain == "vanse.vancitylandscaper.com":
        results = verify_outreach_subdomain()
    elif domain == "vancitylandscaper.com":
        results = verify_root_domain()
    elif domain == "vanseindustry.com":
        results = verify_business_domain()
    else:
        results = []

    click.echo(f"\n{'='*60}")
    click.echo(f"  DNS Verification — {domain}")
    click.echo(f"{'='*60}\n")

    failures = 0
    for r in results:
        symbol = _status_symbol(r.status)
        click.echo(f"  [{symbol}] {r.name}: {r.message}")
        for d in r.details:
            click.echo(f"         {d}")
        if r.status == Status.FAIL:
            failures += 1

    click.echo(f"\n{'='*60}")
    total = len(results)
    passed = sum(1 for r in results if r.status == Status.PASS)
    warned = sum(1 for r in results if r.status == Status.WARN)
    click.echo(f"  {passed}/{total} passed, {warned} warnings, {failures} failures")
    click.echo(f"{'='*60}\n")

    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
