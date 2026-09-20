"""Deterministic camera commands for public-validation image acquisition.

This policy is an offline data-collection tool.  It is deliberately isolated
from the final inference camera policies: the configured centre is a source
frame tile, not a learned prediction or a validation answer map.
"""
from __future__ import annotations

from dtos import RequestedViewDto


LEVEL1_X = (960, 2880)
LEVEL1_Y = (540, 1620)
LEVEL2_X = (480, 3360)
LEVEL2_Y = (270, 1890)


def validate_native_center(center_x: int, center_y: int) -> tuple[int, int]:
    if not LEVEL2_X[0] <= center_x <= LEVEL2_X[1]:
        raise ValueError(f"native acquisition center_x must be in {LEVEL2_X}")
    if not LEVEL2_Y[0] <= center_y <= LEVEL2_Y[1]:
        raise ValueError(f"native acquisition center_y must be in {LEVEL2_Y}")
    return int(center_x), int(center_y)


def acquisition_view(current_level: int, center_x: int, center_y: int) -> RequestedViewDto:
    """Reach a fixed Level-2 tile through one legal Level-1 transition."""
    center_x, center_y = validate_native_center(center_x, center_y)
    if current_level == 0:
        # A Level-0 response cannot request Level 2.  The clipped intermediate
        # is inside Level-1 bounds and at most 551 px from any valid L2 target.
        return RequestedViewDto(
            resolution_level=1,
            center_x=min(max(center_x, LEVEL1_X[0]), LEVEL1_X[1]),
            center_y=min(max(center_y, LEVEL1_Y[0]), LEVEL1_Y[1]),
        )
    if current_level in (1, 2):
        return RequestedViewDto(
            resolution_level=2,
            center_x=center_x,
            center_y=center_y,
        )
    raise ValueError(f"unknown resolution level {current_level}")
