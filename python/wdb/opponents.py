"""
python/wdb/opponents.py

Standalone opponent policies.
Role: Co-Driver.
Ownership boundary: This module owns isolated opponent mechanics only.
It does not modify the core PPO pipeline or training loops.
"""

from math import hypot


class ScriptedChaser:
    """Safety-filtered opponent that avoids hazards before pursuing Player 1."""

    EDGE_CENTER_LIMIT = 8.0
    EDGE_MARGIN = 2.0
    BALL_DANGER_RADIUS = 2.5
    GRAPPLE_RANGE = 5.0
    HEAVY_RANGE = 3.0

    HORIZONTAL_DEADZONE = 0.25
    VERTICAL_DEADZONE = 0.5

    def _get_directions(self, dx: float, dy: float) -> tuple[int, int]:
        """Convert a desired movement vector into movement and jump actions."""

        if dx < -self.HORIZONTAL_DEADZONE:
            move_dir = 0
        elif dx > self.HORIZONTAL_DEADZONE:
            move_dir = 2
        else:
            move_dir = 1

        # Assumes larger y coordinates represent higher positions.
        if dy < -self.VERTICAL_DEADZONE:
            jump_dir = 0
        elif dy > self.VERTICAL_DEADZONE:
            jump_dir = 2
        else:
            jump_dir = 1

        return move_dir, jump_dir

    def get_action(self, state_dict) -> list[int]:
        """
        Return [move_dir, jump_dir, heavy, grapple].

        Safety priority:
        1. Retreat toward map center when near the edge.
        2. Retreat from the ball when it is within 2.5 units.
        3. Pursue Player 1 when neither hazard is present.

        Grapple activates only during pursuit and within 5 units of P1.
        """

        p1 = state_dict["p1"]
        p2 = state_dict["p2"]
        ball = state_dict["ball"]

        center_x = float(state_dict["map_center_x"])
        center_y = float(state_dict["map_center_y"])
        map_radius = float(state_dict["map_radius"])

        p2_x = float(p2["x"])
        p2_y = float(p2["y"])

        # --------------------------------------------------
        # 1. Map-edge safety check
        # --------------------------------------------------
        center_dx = center_x - p2_x
        center_dy = center_y - p2_y

        center_distance = hypot(center_dx, center_dy)
        distance_to_edge = map_radius - center_distance

        near_edge = (
            center_distance > self.EDGE_CENTER_LIMIT
            or distance_to_edge <= self.EDGE_MARGIN
        )

        if near_edge:
            move_dir, jump_dir = self._get_directions(
                center_dx, center_dy
            )
            return [move_dir, jump_dir, 0, 0]

        # --------------------------------------------------
        # 2. Ball avoidance check
        # --------------------------------------------------
        ball_dx = p2_x - float(ball["x"])
        ball_dy = p2_y - float(ball["y"])

        ball_distance = hypot(ball_dx, ball_dy)

        if ball_distance < self.BALL_DANGER_RADIUS:
            # Move directly away from the ball.
            # Handle overlapping positions with a deterministic fallback.
            if ball_distance == 0.0:
                ball_dx = -float(ball.get("vx", 0.0))
                ball_dy = -float(ball.get("vy", 0.0))

                if ball_dx == 0.0 and ball_dy == 0.0:
                    ball_dx = p2_x - center_x
                    ball_dy = p2_y - center_y

                if ball_dx == 0.0 and ball_dy == 0.0:
                    ball_dx = -1.0

            move_dir, jump_dir = self._get_directions(
                ball_dx, ball_dy
            )
            return [move_dir, jump_dir, 0, 0]

        # --------------------------------------------------
        # 3. Safe: pursue Player 1
        # --------------------------------------------------
        p1_dx = float(p1["x"]) - p2_x
        p1_dy = float(p1["y"]) - p2_y

        p1_distance = hypot(p1_dx, p1_dy)

        move_dir, jump_dir = self._get_directions(
            p1_dx, p1_dy
        )

        # Attack actions are enabled only in the safe pursuit branch.
        heavy = int(p1_distance <= self.HEAVY_RANGE)
        grapple = int(p1_distance < self.GRAPPLE_RANGE)

        return [move_dir, jump_dir, heavy, grapple]