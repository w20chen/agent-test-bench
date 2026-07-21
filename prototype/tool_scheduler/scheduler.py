"""Scheduling decision engine.

Generates candidate placements and evaluates them using the cost model.
Produces KEEP / RECOMMEND_MOVE decisions based on configurable thresholds.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .topology import (
    Topology,
    discover as discover_topology,
    get_current_cpu_for_pid,
    get_current_numa_node,
)
from .cost_model import (
    Placement,
    CostBreakdown,
    CostModelConfig,
    compute_total_cost,
)
from .predictor import Predictor
from .idle import idle_breakdown
from .bandwidth import get_bandwidth_utilization

logger = logging.getLogger(__name__)

# Minimum runtime before scheduling decisions (seconds)
MIN_RUNTIME_BEFORE_DECISION = 2.0

# Cooldown between consecutive recommendations (seconds)
DEFAULT_COOLDOWN_SECONDS = 5.0


@dataclass
class Decision:
    """A single scheduling decision."""

    elapsed_s: float
    stable: bool
    predicted_cores: float
    requested_cores: int
    memory_sensitivity: str

    current_cost: Optional[float]
    best_cost: Optional[float]
    gain: Optional[float]
    action: str  # "keep", "recommend_move", "unstable"
    recommended_placement: Optional[str]

    current_cost_breakdown: Optional[dict] = None
    best_cost_breakdown: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict = {
            "elapsed_s": round(self.elapsed_s, 2),
            "stable": self.stable,
            "predicted_cores": round(self.predicted_cores, 1),
            "requested_cores": self.requested_cores,
            "memory_sensitivity": self.memory_sensitivity,
            "action": self.action,
        }
        if self.current_cost is not None:
            d["current_cost"] = round(self.current_cost, 3)
        if self.best_cost is not None:
            d["best_cost"] = round(self.best_cost, 3)
        if self.gain is not None:
            d["gain"] = round(self.gain, 3)
        if self.recommended_placement is not None:
            d["recommended_placement"] = self.recommended_placement
        if self.current_cost_breakdown is not None:
            d["current_cost_breakdown"] = self.current_cost_breakdown
        if self.best_cost_breakdown is not None:
            d["best_cost_breakdown"] = self.best_cost_breakdown
        return d


class Scheduler:
    """Generates scheduling decisions for a tool invocation.

    Evaluates candidate CPU placements and recommends moves based on
    predicted core demand, memory sensitivity, and hardware topology.
    """

    def __init__(
        self,
        predictor: Predictor,
        topology: Topology,
        history: dict[str, dict],
        cost_config: Optional[CostModelConfig] = None,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        command_sig: str = "",
    ) -> None:
        self._predictor = predictor
        self._topology = topology
        self._history = history
        self._cost_config = cost_config or CostModelConfig()
        self._cooldown_seconds = cooldown_seconds
        self._command_sig = command_sig

        self._last_decision_time: float = -float("inf")
        self._decisions: list[Decision] = []
        self._current_placement: Optional[Placement] = None
        self._was_unstable: bool = False

    @property
    def decisions(self) -> list[Decision]:
        return list(self._decisions)

    def _get_memory_sensitivity(self) -> str:
        """Get memory sensitivity from history or default to unknown."""
        hist = self._history.get(self._command_sig, {})
        return hist.get("memory_sensitivity", "unknown")

    def _generate_candidates(
        self,
        root_pid: int,
    ) -> tuple[Optional[Placement], list[Placement]]:
        """Generate candidate placements on the same NUMA node.

        Uses real /proc/stat idle detection and PMU bandwidth info
        when available.  Falls back to "all cores available" on
        non-Linux or first sample.

        Returns:
            (current_placement, list_of_candidate_placements)
        """
        if not self._topology.available:
            return None, []

        # Determine current NUMA node
        current_numa = get_current_numa_node(root_pid)
        if current_numa is None:
            # Default to NUMA node 0
            current_numa = self._topology.numa_nodes[0] if self._topology.numa_nodes else 0

        # Get LLC groups within this NUMA node
        llc_groups = self._topology.llc_groups.get(current_numa, {})
        if not llc_groups:
            return None, []

        # Get real bandwidth utilization for this NUMA node
        bw_util = get_bandwidth_utilization(current_numa)

        candidates: list[Placement] = []
        current_placement: Optional[Placement] = None

        for llc_id, cpu_list in sorted(llc_groups.items()):
            # Use real idle detection from /proc/stat
            free_phys, free_smt = idle_breakdown(
                cpu_list,
                physical_cores_per_cpu=self._topology.physical_cores_per_cpu,
            )

            placement_id = f"numa{current_numa}-llc{llc_id}"

            placement = Placement(
                placement_id=placement_id,
                numa_node=current_numa,
                llc_group=llc_id,
                cpus=sorted(cpu_list),
                available_physical_cores=free_phys,
                available_smt_threads=free_smt,
                # Real bandwidth utilization from PMU, or None if unavailable
                bandwidth_utilization=bw_util,
            )
            candidates.append(placement)

        # Determine current placement by checking which LLC group the
        # process is actually running on (via /proc/<pid>/stat).
        current_placement: Optional[Placement] = None
        current_cpu = get_current_cpu_for_pid(root_pid)
        if current_cpu is not None:
            for candidate in candidates:
                if current_cpu in candidate.cpus:
                    current_placement = candidate
                    break
        # Fallback: if the CPU lookup fails or the CPU isn't in any
        # candidate (race with reschedule), assume the first candidate.
        if current_placement is None and candidates:
            current_placement = candidates[0]

        return current_placement, candidates

    def evaluate(
        self,
        elapsed_s: float,
        root_pid: int,
    ) -> Optional[Decision]:
        """Evaluate whether to generate a scheduling decision.

        Args:
            elapsed_s: Seconds since tool start.
            root_pid: Root process PID.

        Returns:
            A Decision if one should be made, or None to skip.
        """
        # Minimum runtime gate
        if elapsed_s < MIN_RUNTIME_BEFORE_DECISION:
            return None

        # Cooldown gate
        if elapsed_s - self._last_decision_time < self._cooldown_seconds:
            return None

        # Stability gate
        if not self._predictor.stable:
            return None

        predicted_cores = self._predictor.predicted_cores
        requested_cores = self._predictor.requested_cores
        memory_sensitivity = self._get_memory_sensitivity()

        # Generate candidates
        current_placement, candidates = self._generate_candidates(root_pid)

        if not candidates:
            return None

        # Evaluate all candidates
        best_cost: Optional[float] = None
        best_breakdown: Optional[CostBreakdown] = None
        best_candidate: Optional[Placement] = None

        current_breakdown: Optional[CostBreakdown] = None
        current_cost: Optional[float] = None

        for candidate in candidates:
            breakdown = compute_total_cost(
                predicted_cores=predicted_cores,
                memory_sensitivity=memory_sensitivity,
                candidate=candidate,
                current_placement=current_placement,
                config=self._cost_config,
            )

            # Skip infeasible (cross-NUMA) placements
            if breakdown.total_cost >= float("inf"):
                continue

            # Check if this is the current placement
            if (
                current_placement is not None
                and candidate.placement_id == current_placement.placement_id
            ):
                current_cost = breakdown.total_cost
                current_breakdown = breakdown

            if best_cost is None or breakdown.total_cost < best_cost:
                best_cost = breakdown.total_cost
                best_breakdown = breakdown
                best_candidate = candidate

        if best_cost is None or best_breakdown is None:
            return None

        # Compute gain
        if current_cost is None:
            # No current placement identified; use the first candidate as baseline
            current_cost = best_cost
            current_breakdown = best_breakdown

        gain = current_cost - best_cost
        action = "keep"

        if gain > self._cost_config.migration_threshold:
            action = "recommend_move"

        decision = Decision(
            elapsed_s=elapsed_s,
            stable=True,
            predicted_cores=predicted_cores,
            requested_cores=requested_cores,
            memory_sensitivity=memory_sensitivity,
            current_cost=current_cost,
            best_cost=best_cost,
            gain=gain,
            action=action,
            recommended_placement=(
                best_candidate.placement_id if best_candidate else None
            ),
            current_cost_breakdown=(
                current_breakdown.to_dict() if current_breakdown else None
            ),
            best_cost_breakdown=(
                best_breakdown.to_dict() if best_breakdown else None
            ),
        )

        self._last_decision_time = elapsed_s
        self._decisions.append(decision)
        self._current_placement = best_candidate

        return decision

    def check_phase_change(self) -> bool:
        """Check if the predictor has diverged, indicating a phase change."""
        return self._predictor.check_divergence()

    def reset_after_phase_change(self) -> None:
        """Reset stability tracking after phase change."""
        self._predictor.reset_stability()
        self._was_unstable = True
