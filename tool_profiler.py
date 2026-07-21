#!/usr/bin/env python3
"""Convenience entry point: python tool_profiler.py -- <command>

Equivalent to: python -m prototype.tool_profiler -- <command>
"""

import sys
from prototype.tool_profiler.cli import main

if __name__ == "__main__":
    main()
