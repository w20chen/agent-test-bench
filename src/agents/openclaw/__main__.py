"""CLI entry point for OpenClaw agent.

Usage: python -m agents.openclaw --prompt "..." --workspace /path/to/dir
"""

from agents.openclaw._cli import main

if __name__ == "__main__":
    main()
