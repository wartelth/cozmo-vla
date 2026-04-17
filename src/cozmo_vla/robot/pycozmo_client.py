"""PyCozmo connection, camera, proprioception, and normalized motor commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

try:
    import pycozmo
    from pycozmo import robot as cozmo_robot
except ImportError as e:  # pragma: no cover
    pycozmo = None  # type: ignore[misc, assignment]
    cozmo_robot = None  # type: ignore[misc, assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _ensure_pycozmo():
    if pycozmo is None:
        raise ImportError(
            "pycozmo is required for the robot client. Install with: pip install pycozmo"
        ) from _IMPORT_ERROR


@dataclass
class SafetyConfig:
    """Stop motors when unsafe conditions are reported."""

    stop_on_cliff: bool = True
    stop_on_picked_up: bool = True


class PyCozmoRobot:
    """
    Wraps ``pycozmo.Client`` with:
    - latest RGB frame from ``EvtNewRawCameraImage``
    - 8-D ``observation.state`` matching ``configs/cozmo_action_space.json``
    - action vector in [-1, 1] mapped to wheel speeds (mm/s) and lift/head velocities
    """

    def __init__(self, cli: object, safety: SafetyConfig | None = None):
        _ensure_pycozmo()
        self._cli = cli
        self.safety = safety or SafetyConfig()
        self._last_image: Image | None = None
        self._unsafe_latched = False
        self._cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self._on_camera_image)

    def _on_camera_image(self, cli: object, image: "Image") -> None:
        del cli
        self._last_image = image

    def enable_camera(self, *, color: bool = True) -> None:
        self._cli.enable_camera(enable=True, color=color)

    def get_rgb_uint8(self) -> np.ndarray:
        """Return HxWx3 RGB uint8; blocks briefly if no frame yet."""
        import time

        for _ in range(300):  # ~3s at 10ms
            if self._last_image is not None:
                arr = np.asarray(self._last_image.convert("RGB"), dtype=np.uint8)
                return arr
            time.sleep(0.01)
        raise TimeoutError("No camera frame received yet. Is the camera enabled?")

    def _head_angle_norm(self) -> float:
        mn = cozmo_robot.MIN_HEAD_ANGLE.radians
        mx = cozmo_robot.MAX_HEAD_ANGLE.radians
        h = self._cli.head_angle.radians
        return float(np.clip(2.0 * (h - mn) / (mx - mn) - 1.0, -1.0, 1.0))

    def _lift_ratio(self) -> float:
        return float(self._cli.lift_position.ratio)

    def _wheel_norm(self, speed_mmps: float) -> float:
        max_sp = float(cozmo_robot.MAX_WHEEL_SPEED.mmps)
        return float(np.clip(speed_mmps / max_sp, -1.0, 1.0))

    def get_state_vector(self) -> np.ndarray:
        """8-float proprioception vector (matches dataset)."""
        cli = self._cli
        pose_yaw = float(cli.pose.rotation.angle_z.radians)
        # Normalize yaw to [-1, 1] roughly
        yaw_norm = float(np.clip(pose_yaw / np.pi, -1.0, 1.0))
        accel_z = float(cli.accel.z)
        accel_z_norm = float(np.clip(accel_z / 20.0, -1.0, 1.0))

        cliff = 1.0 if (cli.robot_status & cozmo_robot.RobotStatusFlag.CLIFF_DETECTED) else 0.0
        picked = 1.0 if (cli.robot_status & cozmo_robot.RobotStatusFlag.IS_PICKED_UP) else 0.0

        return np.array(
            [
                self._lift_ratio(),
                self._head_angle_norm(),
                self._wheel_norm(cli.left_wheel_speed.mmps),
                self._wheel_norm(cli.right_wheel_speed.mmps),
                yaw_norm,
                accel_z_norm,
                0.0,
                max(cliff, picked) if self.safety.stop_on_picked_up else cliff,
            ],
            dtype=np.float32,
        )

    def is_unsafe(self) -> bool:
        st = self._cli.robot_status
        if self.safety.stop_on_cliff and (st & cozmo_robot.RobotStatusFlag.CLIFF_DETECTED):
            return True
        if self.safety.stop_on_picked_up and (st & cozmo_robot.RobotStatusFlag.IS_PICKED_UP):
            return True
        return False

    def apply_action_normalized(self, action: np.ndarray, *, stop_if_unsafe: bool = True) -> None:
        """
        ``action`` length 4: [left_wheel, right_wheel, lift_vel, head_vel] in [-1, 1].
        """
        if stop_if_unsafe and self.is_unsafe():
            self._cli.stop_all_motors()
            if not self._unsafe_latched:
                self._unsafe_latched = True
                logger.warning("Safety stop: cliff or picked-up flag set.")
            return

        self._unsafe_latched = False
        lw = float(np.clip(action[0], -1.0, 1.0))
        rw = float(np.clip(action[1], -1.0, 1.0))
        lift_v = float(np.clip(action[2], -1.0, 1.0))
        head_v = float(np.clip(action[3], -1.0, 1.0))

        max_mmps = float(cozmo_robot.MAX_WHEEL_SPEED.mmps)
        self._cli.drive_wheels(
            int(lw * max_mmps),
            int(rw * max_mmps),
        )
        # RC example uses ~0.8 max for lift; scale [-1,1] -> rad/s request
        self._cli.move_lift(lift_v * 1.0)
        self._cli.move_head(head_v * 1.0)

    def stop(self) -> None:
        self._cli.stop_all_motors()

    def get_debug_info(self) -> dict[str, float | int | str | bool]:
        """Readable snapshot for teleop / diagnostics (no extra hardware calls)."""
        cli = self._cli
        st = int(cli.robot_status)
        cliff = bool(st & cozmo_robot.RobotStatusFlag.CLIFF_DETECTED)
        picked = bool(st & cozmo_robot.RobotStatusFlag.IS_PICKED_UP)
        cam_shape = "-"
        if self._last_image is not None:
            im = self._last_image
            w, h = getattr(im, "size", (im.width, im.height))
            cam_shape = f"{w}x{h}"
        return {
            "battery_v": float(cli.battery_voltage),
            "left_wheel_mmps": float(cli.left_wheel_speed.mmps),
            "right_wheel_mmps": float(cli.right_wheel_speed.mmps),
            "head_deg": float(cli.head_angle.degrees),
            "lift_ratio": float(cli.lift_position.ratio),
            "robot_status": st,
            "robot_status_hex": hex(st),
            "cliff": cliff,
            "picked_up": picked,
            "unsafe": self.is_unsafe(),
            "camera_shape": cam_shape,
            "accel_z": float(cli.accel.z),
        }


def run_with_robot(fn: Callable[[PyCozmoRobot], None], **connect_kwargs) -> None:
    """Run ``fn(robot)`` inside ``pycozmo.connect()``."""
    _ensure_pycozmo()
    with pycozmo.connect(**connect_kwargs) as cli:
        robot = PyCozmoRobot(cli)
        angle = (cozmo_robot.MAX_HEAD_ANGLE.radians - cozmo_robot.MIN_HEAD_ANGLE.radians) * 0.25
        cli.set_head_angle(angle)
        robot.enable_camera(color=True)
        fn(robot)
