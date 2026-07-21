#!/usr/bin/env python3
"""Convenience entry point: python tool_scheduler.py --dry-run -- <command>

Equivalent to: python -m prototype.tool_scheduler --dry-run -- <command>
"""

import sys
from prototype.tool_scheduler.cli import main

if __name__ == "__main__":
    main()
