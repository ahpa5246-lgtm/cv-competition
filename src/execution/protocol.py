"""Execution contract shared by simulator and future robot adapters."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from src.core.types import ExecutionResult, RobotTrajectory


class VisualEnvironment(Protocol):
    def render(self) -> np.ndarray: ...

    def execute(self, trajectory: RobotTrajectory) -> ExecutionResult: ...
