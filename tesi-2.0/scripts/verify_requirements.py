#!/usr/bin/env python3
"""Verify requirements.txt against installed package versions.

Parses requirements.txt, compares each package version with installed version
from importlib.metadata, outputs table report.

Exit code: 0 if all match, 1 if any mismatch.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements.txt"


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    """Parse requirements.txt, skip comments and blank lines.

    Returns list of (package_name, pinned_version).
    """
    packages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse "package==version"
            match = re.match(r"^([a-zA-Z0-9\-_.]+)==(.+)$", line)
            if match:
                pkg, ver = match.groups()
                packages.append((pkg, ver))
    return packages


def get_installed_version(package_name: str) -> str | None:
    """Get installed version via importlib.metadata, return None if not found."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    """Main verify logic."""
    if not REQUIREMENTS_PATH.exists():
        print(f"Error: {REQUIREMENTS_PATH} not found", file=sys.stderr)
        return 1

    packages = parse_requirements(REQUIREMENTS_PATH)
    if not packages:
        print("No packages found in requirements.txt", file=sys.stderr)
        return 1

    print(f"Verifying {len(packages)} packages...\n")
    print(f"{'Package':<30} {'Required':<15} {'Installed':<15} {'Match':<6}")
    print("-" * 70)

    mismatches = []
    for pkg_name, req_ver in packages:
        inst_ver = get_installed_version(pkg_name)
        if inst_ver is None:
            match = "✗ (not installed)"
            mismatches.append((pkg_name, req_ver, "NOT_INSTALLED"))
        elif inst_ver == req_ver:
            match = "✓"
        else:
            match = "✗"
            mismatches.append((pkg_name, req_ver, inst_ver))

        inst_display = inst_ver or "—"
        print(f"{pkg_name:<30} {req_ver:<15} {inst_display:<15} {match:<6}")

    print("-" * 70)
    print(f"Total: {len(packages)} packages. Matches: {len(packages) - len(mismatches)}/{len(packages)}")

    if mismatches:
        print(f"\n⚠ {len(mismatches)} MISMATCH(ES):")
        for pkg, req, inst in mismatches:
            inst_str = inst if inst != "NOT_INSTALLED" else "not installed"
            print(f"  {pkg}: required={req}, installed={inst_str}")
        return 1

    print("\n✓ All packages match!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
