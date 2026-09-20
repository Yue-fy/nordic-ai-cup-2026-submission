"""Account for commands whose effect has not appeared in a received view yet."""
import math
from dtos import ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS


class CameraCommandGuard:
    def __init__(self):
        self.pending = []

    def filter(self, request, command):
        current = (request.view.resolution_level, request.view.center_x, request.view.center_y)
        # A received view is our only acknowledgement. Do not infer the camera
        # position from the sequence frame number or a known scene route.
        matches = [i for i, target in enumerate(self.pending) if target == current]
        if matches:
            self.pending = self.pending[max(matches)+1:]
        feedback = request.camera_command_feedback
        if feedback is not None:
            rejected = feedback.requested_view
            key = (rejected.resolution_level, rejected.center_x, rejected.center_y)
            self.pending = [target for target in self.pending if target != key]
        if command is None:
            return None
        target = (command.resolution_level, command.center_x, command.center_y)
        for level, x, y in [current, *self.pending]:
            if target[0] not in ALLOWED_RESOLUTION_LEVELS[level]:
                return None
            full_reset = target[0] == 0 and request.camera_constraints.full_view_reset_exempt_from_delta
            limit = MAXIMUM_CENTER_DELTA_PIXELS[level]
            if (level,x,y)==current:
                limit = min(limit, request.camera_constraints.maximum_center_delta)
            if not full_reset and math.hypot(target[1]-x,target[2]-y)>limit:
                return None
        if target not in self.pending and target != current:
            self.pending.append(target)
        return command
