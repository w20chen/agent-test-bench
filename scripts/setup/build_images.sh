#!/usr/bin/env bash
# Container image building is no longer required.
# LocalSandbox uses proot + temp directories for task isolation.
echo "[setup] build_images: skipped — LocalSandbox requires no container image"
exit 0
