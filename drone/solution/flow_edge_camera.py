"""Camera coverage focused on the edge where image-derived new terrain enters."""
import numpy as np

from dtos import RequestedViewDto


class FlowEdgeCamera:
    def __init__(self, wide_interval=8, focus_after_wide=False):
        if wide_interval < 2:
            raise ValueError("wide_interval must be at least 2")
        self.steps = 0
        self.wide_interval = wide_interval
        self.startup_cursor = 0
        self.edge_cursor = 0
        self.fallback_cursor = 0
        self.motion_history = []
        self.last_mode = "startup"
        self.focus_after_wide = focus_after_wide

    @staticmethod
    def _six_targets(bounds):
        left, right = bounds.minimum_center_x, bounds.maximum_center_x
        top, bottom = bounds.minimum_center_y, bounds.maximum_center_y
        middle = (left + right) // 2
        return [(left, top), (middle, top), (right, top),
                (right, bottom), (middle, bottom), (left, bottom)]

    @staticmethod
    def _edge_targets(bounds, motion):
        left, right = bounds.minimum_center_x, bounds.maximum_center_x
        top, bottom = bounds.minimum_center_y, bounds.maximum_center_y
        middle_x = (left + right) // 2
        middle_y = (top + bottom) // 2
        dx, dy = motion
        if abs(dy) >= abs(dx):
            # Ground features moving down reveal new terrain at the top.
            y = top if dy > 0 else bottom
            return [(left, y), (middle_x, y), (right, y)]
        x = left if dx > 0 else right
        return [(x, top), (x, middle_y), (x, bottom)]

    @staticmethod
    def _reachable_target(request, target):
        constraints, view = request.camera_constraints, request.view
        bounds = constraints.bounds_for_level(1)
        target = np.asarray(target, dtype=float)
        target = np.clip(target, [bounds.minimum_center_x, bounds.minimum_center_y],
                         [bounds.maximum_center_x, bounds.maximum_center_y])
        current = np.array([view.center_x, view.center_y], dtype=float)
        delta = target - current
        distance = np.linalg.norm(delta)
        reached = distance <= constraints.maximum_center_delta
        if not reached and distance:
            target = current + delta * (max(constraints.maximum_center_delta - 2, 0) / distance)
        x, y = np.rint(target).astype(int)
        x = int(np.clip(x, bounds.minimum_center_x, bounds.maximum_center_x))
        y = int(np.clip(y, bounds.minimum_center_y, bounds.maximum_center_y))
        if np.hypot(x - view.center_x, y - view.center_y) > constraints.maximum_center_delta:
            return None, False
        return RequestedViewDto(resolution_level=1, center_x=x, center_y=y), reached

    @staticmethod
    def _detection_focus(bounds, motion, annotations, width, height):
        if not annotations:
            return None, None
        left, right = bounds.minimum_center_x, bounds.maximum_center_x
        top, bottom = bounds.minimum_center_y, bounds.maximum_center_y
        dx, dy = motion
        centres = []
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            centres.append(((x1 + x2) * width / 2, (y1 + y2) * height / 2, annotation.confidence))
        if abs(dy) >= abs(dx):
            ingress = top if dy > 0 else bottom
            x, y, _ = min(centres, key=lambda row: (abs(row[1] - ingress), -row[2]))
            target = (int(np.clip(x, left, right)), ingress)
            index = int(np.argmin([abs(target[0] - value) for value in [left, (left + right) // 2, right]]))
        else:
            ingress = left if dx > 0 else right
            x, y, _ = min(centres, key=lambda row: (abs(row[0] - ingress), -row[2]))
            target = (ingress, int(np.clip(y, top, bottom)))
            index = int(np.argmin([abs(target[1] - value) for value in [top, (top + bottom) // 2, bottom]]))
        return target, index

    def choose(self, request, motion, quality, annotations=None):
        constraints = request.camera_constraints
        self.steps += 1
        motion = np.asarray(motion, dtype=float).reshape(2)
        if quality >= 12 and np.isfinite(motion).all() and 2 <= np.linalg.norm(motion) <= 500:
            self.motion_history.append(motion.copy())
            self.motion_history = self.motion_history[-8:]
        if self.steps % self.wide_interval == 0 and 0 in constraints.allowed_resolution_levels:
            bounds = constraints.bounds_for_level(0)
            self.last_mode = "wide"
            return RequestedViewDto(resolution_level=0, center_x=int(bounds.minimum_center_x),
                                    center_y=int(bounds.minimum_center_y))
        bounds = constraints.bounds_for_level(1)
        if bounds is None:
            return None
        six = self._six_targets(bounds)
        if self.startup_cursor < len(six):
            target = six[self.startup_cursor]
            command, reached = self._reachable_target(request, target)
            if reached:
                self.startup_cursor += 1
            self.last_mode = "startup"
            return command
        if self.motion_history:
            direction = np.median(np.stack(self.motion_history), axis=0)
            targets = self._edge_targets(bounds, direction)
            if self.focus_after_wide and request.view.resolution_level == 0:
                target, index = self._detection_focus(bounds, direction, annotations,
                                                      request.original_width, request.original_height)
                if target is not None:
                    # Continue from the focused tile toward the middle, then
                    # traverse the edge without a long endpoint wrap.
                    self.edge_cursor = {0: 1, 1: 2, 2: 3}[index]
                    command, _ = self._reachable_target(request, target)
                    self.last_mode = "focus"
                    return command
            order = [0, 1, 2, 1] if self.focus_after_wide else [0, 1, 2]
            target = targets[order[self.edge_cursor % len(order)]]
            command, reached = self._reachable_target(request, target)
            if reached:
                self.edge_cursor += 1
            self.last_mode = "edge"
            return command
        target = six[self.fallback_cursor % len(six)]
        command, reached = self._reachable_target(request, target)
        if reached:
            self.fallback_cursor += 1
        self.last_mode = "fallback"
        return command
