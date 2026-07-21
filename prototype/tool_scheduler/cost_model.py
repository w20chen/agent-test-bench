"""Simplified cost model for CPU placement decisions.

Computes total_cost = core_cost + memory_cost + move_cost for each
candidate placement, enabling dry-run scheduling recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

logger = __import__("logging").getLogger(__name__)


# ---- Configurable parameters ----

# SMT thread weight: an SMT sibling counts as 0.3 of a physical core
DEFAULT_SMT_WEIGHT = 0.30

# Unknown memory sensitivity: weight applied to bandwidth_utilization
DEFAULT_UNKNOWN_MEMORY_WEIGHT = 0.30

# Move costs (relative constants)
MOVE_COST_SAME_PLACEMENT = 0.00
MOVE_COST_SAME_LLC = 0.05
MOVE_COST_SAME_NUMA = 0.15
MOVE_COST_CROSS_NUMA = float("inf")  # Never generate cross-NUMA candidates

# Migration threshold: minimum gain required to recommend a move
DEFAULT_MIGRATION_THRESHOLD = 0.20

MemorySensitivity = str  # "high" | "low" | "unknown"


@dataclass
class Placement:
    """A candidate CPU placement."""

    placement_id: str
    numa_node: int
    llc_group: int
    cpus: list[int]
    available_physical_cores: int
    available_smt_threads: int
    bandwidth_utilization: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "placement_id": self.placement_id,
            "numa_node": self.numa_node,
            "llc_group": self.llc_group,
            "cpus": self.cpus,
            "available_physical_cores": self.available_physical_cores,
            "available_smt_threads": self.available_smt_threads,
            "bandwidth_utilization": self.bandwidth_utilization,
        }


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a placement."""

    placement_id: str
    core_cost: float
    memory_cost: float
    move_cost: float
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "placement": self.placement_id,
            "core_cost": round(self.core_cost, 3),
            "memory_cost": round(self.memory_cost, 3),
            "move_cost": round(self.move_cost, 3),
            "total_cost": round(self.total_cost, 3),
        }


@dataclass
class CostModelConfig:
    """Configuration for the cost model."""

    smt_weight: float = DEFAULT_SMT_WEIGHT
    unknown_memory_weight: float = DEFAULT_UNKNOWN_MEMORY_WEIGHT
    move_cost_same_llc: float = MOVE_COST_SAME_LLC
    move_cost_same_numa: float = MOVE_COST_SAME_NUMA
    migration_threshold: float = DEFAULT_MIGRATION_THRESHOLD

    @classmethod
    def from_dict(cls, d: dict) -> "CostModelConfig":
        return cls(
            smt_weight=float(d.get("smt_weight", DEFAULT_SMT_WEIGHT)),
            unknown_memory_weight=float(
                d.get("unknown_memory_weight", DEFAULT_UNKNOWN_MEMORY_WEIGHT)
            ),
            move_cost_same_llc=float(
                d.get("move_cost_same_llc", MOVE_COST_SAME_LLC)
            ),
            move_cost_same_numa=float(
                d.get("move_cost_same_numa", MOVE_COST_SAME_NUMA)
            ),
            migration_threshold=float(
                d.get("migration_threshold", DEFAULT_MIGRATION_THRESHOLD)
            ),
        )


def compute_effective_available_cores(
    physical_cores: int,
    smt_threads: int,
    smt_weight: float = DEFAULT_SMT_WEIGHT,
) -> float:
    """Compute effective available cores accounting for SMT.

    Args:
        physical_cores: Number of available physical cores.
        smt_threads: Number of available SMT threads (siblings of
            already-allocated physical cores).
        smt_weight: Weight for SMT threads relative to physical cores.

    Returns:
        Effective available core count as a float.
    """
    return float(physical_cores) + smt_weight * float(smt_threads)


def compute_core_cost(
    predicted_cores: float,
    available_physical_cores: int,
    available_smt_threads: int = 0,
    smt_weight: float = DEFAULT_SMT_WEIGHT,
) -> float:
    """Compute core shortage cost.

    Args:
        predicted_cores: Tool's predicted CPU core demand.
        available_physical_cores: Number of physical cores in the candidate.
        available_smt_threads: Number of SMT threads in the candidate.
        smt_weight: Weight applied to SMT threads.

    Returns:
        Core cost in [0, 1]. 0 = fully satisfied, higher = more shortage.
    """
    effective = compute_effective_available_cores(
        available_physical_cores,
        available_smt_threads,
        smt_weight,
    )
    max_cores = max(predicted_cores, 1.0)
    return max(0.0, (predicted_cores - effective) / max_cores)


def compute_memory_cost(
    memory_sensitivity: MemorySensitivity,
    bandwidth_utilization: Optional[float],
    unknown_memory_weight: float = DEFAULT_UNKNOWN_MEMORY_WEIGHT,
) -> float:
    """Compute memory contention cost.

    Args:
        memory_sensitivity: "high", "low", or "unknown".
        bandwidth_utilization: Current bandwidth utilization [0,1] or None.
        unknown_memory_weight: Weight applied when sensitivity is unknown.

    Returns:
        Memory cost in [0, 1].
    """
    if bandwidth_utilization is None:
        return 0.0

    if memory_sensitivity == "high":
        return bandwidth_utilization
    elif memory_sensitivity == "low":
        return 0.0
    else:  # "unknown"
        return unknown_memory_weight * bandwidth_utilization


def compute_move_cost(
    current_placement: Optional[Placement],
    candidate: Placement,
    move_cost_same_llc: float = MOVE_COST_SAME_LLC,
    move_cost_same_numa: float = MOVE_COST_SAME_NUMA,
) -> float:
    """Compute cost of moving from current to candidate placement.

    Args:
        current_placement: The tool's current placement (None if unknown).
        candidate: The candidate placement.
        move_cost_same_llc: Cost of moving within the same LLC.
        move_cost_same_numa: Cost of moving across LLCs within the same NUMA.

    Returns:
        Move cost. Returns inf if cross-NUMA.
    """
    if current_placement is None:
        return 0.0  # No current placement, assume no move cost

    if current_placement.numa_node != candidate.numa_node:
        return MOVE_COST_CROSS_NUMA

    if current_placement.placement_id == candidate.placement_id:
        return MOVE_COST_SAME_PLACEMENT

    if current_placement.llc_group == candidate.llc_group:
        return move_cost_same_llc

    return move_cost_same_numa


def compute_total_cost(
    predicted_cores: float,
    memory_sensitivity: MemorySensitivity,
    candidate: Placement,
    current_placement: Optional[Placement] = None,
    config: Optional[CostModelConfig] = None,
) -> CostBreakdown:
    """Compute the total cost for a candidate placement.

    Args:
        predicted_cores: Tool's predicted CPU core demand.
        memory_sensitivity: Tool's memory sensitivity.
        candidate: The candidate placement to evaluate.
        current_placement: The tool's current placement (for move cost).
        config: Cost model configuration.

    Returns:
        CostBreakdown with detailed costs.
    """
    if config is None:
        config = CostModelConfig()

    core_cost = compute_core_cost(
        predicted_cores,
        candidate.available_physical_cores,
        candidate.available_smt_threads,
        config.smt_weight,
    )

    memory_cost = compute_memory_cost(
        memory_sensitivity,
        candidate.bandwidth_utilization,
        config.unknown_memory_weight,
    )

    move_cost = compute_move_cost(
        current_placement,
        candidate,
        config.move_cost_same_llc,
        config.move_cost_same_numa,
    )

    total = core_cost + memory_cost + move_cost

    return CostBreakdown(
        placement_id=candidate.placement_id,
        core_cost=core_cost,
        memory_cost=memory_cost,
        move_cost=move_cost,
        total_cost=total,
    )
