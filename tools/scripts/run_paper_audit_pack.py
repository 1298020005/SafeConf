#!/usr/bin/env python3
"""Rebuild the 2026-09-06 paper-audit pack from official CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "safeconf_audit"))

from safeconf_audit.paper_pack import main

if __name__ == "__main__":
    raise SystemExit(main())
