"""Low-frequency L2 inspections inserted into the fixed L1 coverage route."""
import numpy as np

from active_detector import CoverageCamera
from dtos import RequestedViewDto, SOURCE_REGION_SIZES


class HybridDetailCamera:
    def __init__(self, cooldown=24, warmup_coverage_steps=8, wide_interval=8):
        if cooldown < 3:
            raise ValueError("cooldown must allow a legal L2 return")
        self.coverage = CoverageCamera(wide_interval)
        self.cooldown = cooldown
        self.warmup_coverage_steps = warmup_coverage_steps
        self.tick = 0
        self.last_detail_tick = -cooldown
        self.returning = False

    @staticmethod
    def command(request, level, centre):
        constraints, view = request.camera_constraints, request.view
        if level not in constraints.allowed_resolution_levels:
            return None
        bounds = constraints.bounds_for_level(level)
        if bounds is None:
            return None
        target = np.clip(np.asarray(centre, float),
                         [bounds.minimum_center_x, bounds.minimum_center_y],
                         [bounds.maximum_center_x, bounds.maximum_center_y])
        current = np.array([view.center_x, view.center_y], float)
        delta = target - current
        distance = float(np.linalg.norm(delta))
        exempt = level == 0 and constraints.full_view_reset_exempt_from_delta
        if not exempt and distance > constraints.maximum_center_delta:
            target = current + delta * (max(0., constraints.maximum_center_delta - 2.) / distance)
        target = np.rint(target).astype(int)
        target = np.clip(target,
                         [bounds.minimum_center_x, bounds.minimum_center_y],
                         [bounds.maximum_center_x, bounds.maximum_center_y])
        if not exempt and np.linalg.norm(target - current) > constraints.maximum_center_delta:
            return None
        return RequestedViewDto(resolution_level=level,
                                center_x=int(target[0]), center_y=int(target[1]))

    @staticmethod
    def eligible_tracks(request, tracks):
        if request.view.resolution_level != 1:
            return []
        bounds = request.camera_constraints.bounds_for_level(2)
        if bounds is None:
            return []
        region = np.asarray(request.view.source_region_xyxy, float)
        output = []
        half = np.asarray(SOURCE_REGION_SIZES[2], float) / 2
        for track in tracks:
            box = np.asarray(track.box, float)
            side = box[2:] - box[:2]
            centre = (box[:2] + box[2:]) / 2
            if track.age > 0 or track.confidence < .08 or np.any(side < 4) or max(side) > 800:
                continue
            if not (region[0] <= centre[0] <= region[2] and region[1] <= centre[1] <= region[3]):
                continue
            target = np.clip(centre,
                             [bounds.minimum_center_x, bounds.minimum_center_y],
                             [bounds.maximum_center_x, bounds.maximum_center_y])
            detail_region = np.r_[target - half, target + half]
            margin = .03 * (detail_region[2:] - detail_region[:2])
            if not (np.all(box[:2] >= detail_region[:2] + margin) and
                    np.all(box[2:] <= detail_region[2:] - margin)):
                continue
            certainty = float(track.evidence.max() / max(track.evidence.sum(), 1e-9))
            # Prefer uncertain and optically small objects; confidence prevents
            # spending camera time on a weak isolated proposal.
            priority = track.confidence * ((1.0 - certainty) + min(1.0, 96.0 / max(min(side), 1.0)))
            output.append((priority, target))
        return output

    def choose(self, request, tracks):
        self.tick += 1
        if self.returning:
            if request.view.resolution_level == 2:
                command = self.command(request, 1, [request.view.center_x, request.view.center_y])
                if command is not None:
                    return command
            self.returning = False

        ready = (self.coverage.steps >= self.warmup_coverage_steps and
                 self.tick - self.last_detail_tick >= self.cooldown)
        if ready:
            candidates = self.eligible_tracks(request, tracks)
            if candidates:
                target = max(candidates, key=lambda item: item[0])[1]
                command = self.command(request, 2, target)
                if command is not None:
                    self.returning = True
                    self.last_detail_tick = self.tick
                    return command

        # Calling the frozen policy only on coverage frames preserves its L1
        # route and periodic L0 resets; inspections merely stretch the route.
        return self.coverage.choose(request)
