"""Keyboard and gamepad teleop providers (shared by collect_data and teleop_debug)."""

from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TeleopProvider(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def action_vector(self) -> np.ndarray: ...
    def consume_episode_done(self) -> bool: ...  # True once = save episode + next instruction prompt


class KeyboardTeleop:
    """Non-blocking keyboard state for normalized actions in [-1, 1]."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._episode_done = False
        self._listener = None

    def _on_press(self, key) -> None:
        try:
            k = key.char.lower()
        except AttributeError:
            return
        with self._lock:
            self._keys.add(k)
            if k == "n":
                self._episode_done = True

    def _on_release(self, key) -> None:
        try:
            k = key.char.lower()
        except AttributeError:
            return
        with self._lock:
            self._keys.discard(k)

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    def consume_episode_done(self) -> bool:
        with self._lock:
            done = self._episode_done
            self._episode_done = False
        return done

    def action_vector(self) -> np.ndarray:
        """Return [lw, rw, lift, head] in [-1, 1]."""
        with self._lock:
            keys = set(self._keys)
        base = 0.0
        turn = 0.0
        if "w" in keys:
            base += 1.0
        if "s" in keys:
            base -= 1.0
        if "a" in keys:
            turn -= 1.0
        if "d" in keys:
            turn += 1.0
        left = float(np.clip(base + turn, -1.0, 1.0))
        right = float(np.clip(base - turn, -1.0, 1.0))

        lift = 0.0
        if "r" in keys:
            lift += 1.0
        if "f" in keys:
            lift -= 1.0

        head = 0.0
        if "t" in keys:
            head += 1.0
        if "g" in keys:
            head -= 1.0

        return np.array([left, right, lift, head], dtype=np.float32)


class GamepadTeleop:
    """
    Xbox-style layout via Pygame (works with Xbox / many XInput pads on Windows).

    - Left stick: forward/back + turn (same mixing as keyboard tank-ish).
    - Right stick Y: lift
    - Right stick X: head
    """

    def __init__(
        self,
        joystick_index: int = 0,
        deadzone: float = 0.15,
        invert_forward: bool = False,
        *,
        next_episode_button: int = 7,
    ) -> None:
        self._joystick_index = joystick_index
        self._deadzone = deadzone
        self._invert_forward = invert_forward
        self._next_episode_button = next_episode_button
        self._next_btn_was_down = False
        self._joy = None

    @staticmethod
    def _dz(v: float, d: float) -> float:
        if abs(v) < d:
            return 0.0
        s = 1.0 if v > 0 else -1.0
        m = (abs(v) - d) / (1.0 - d)
        return float(np.clip(s * m, -1.0, 1.0))

    def start(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= self._joystick_index:
            pygame.quit()
            raise RuntimeError(
                f"No joystick at index {self._joystick_index}. "
                f"Found {pygame.joystick.get_count()} controller(s). Plug in / pair the Xbox controller and retry."
            )
        self._joy = pygame.joystick.Joystick(self._joystick_index)
        self._joy.init()
        self._next_btn_was_down = False
        logging.info("Using controller: %s", self._joy.get_name())
        nbtn = self._joy.get_numbuttons()
        if self._next_episode_button >= nbtn:
            logging.warning(
                "Gamepad has %s buttons; --gamepad-next-button=%s is out of range. "
                "Episode-shortcut button will not work until you lower the index.",
                nbtn,
                self._next_episode_button,
            )

    def stop(self) -> None:
        import pygame

        self._joy = None
        try:
            pygame.joystick.quit()
            pygame.quit()
        except BaseException:
            # Ctrl+C during shutdown can raise KeyboardInterrupt inside pygame
            pass

    def action_vector(self) -> np.ndarray:
        import pygame

        if self._joy is None:
            return np.zeros(4, dtype=np.float32)

        pygame.event.pump()

        def axis(i: int) -> float:
            if i >= self._joy.get_numaxes():
                return 0.0
            return float(self._joy.get_axis(i))

        lx = self._dz(axis(0), self._deadzone)
        ly = self._dz(axis(1), self._deadzone)
        forward = -ly if not self._invert_forward else ly
        turn = lx
        left = float(np.clip(forward + turn, -1.0, 1.0))
        right = float(np.clip(forward - turn, -1.0, 1.0))

        rx = self._dz(axis(3), self._deadzone) if self._joy.get_numaxes() > 3 else 0.0
        ry = self._dz(axis(4), self._deadzone) if self._joy.get_numaxes() > 4 else 0.0
        lift = float(np.clip(ry, -1.0, 1.0))
        head = float(np.clip(rx, -1.0, 1.0))

        return np.array([left, right, lift, head], dtype=np.float32)

    def consume_episode_done(self) -> bool:
        import pygame

        if self._joy is None:
            return False
        pygame.event.pump()
        nbtn = self._joy.get_numbuttons()
        b = self._next_episode_button
        if b < 0 or b >= nbtn:
            return False
        down = bool(self._joy.get_button(b))
        edge = down and not self._next_btn_was_down
        self._next_btn_was_down = down
        return edge

    def raw_axes_snapshot(self) -> list[float]:
        """For debug logging: raw axis values after pygame.event.pump()."""
        import pygame

        if self._joy is None:
            return []
        pygame.event.pump()
        n = self._joy.get_numaxes()
        return [float(self._joy.get_axis(i)) for i in range(n)]


def make_teleop(
    mode: str,
    *,
    joystick_index: int = 0,
    gamepad_deadzone: float = 0.15,
    invert_forward: bool = False,
    gamepad_next_button: int = 7,
) -> TeleopProvider:
    if mode == "keyboard":
        return KeyboardTeleop()
    return GamepadTeleop(
        joystick_index=joystick_index,
        deadzone=gamepad_deadzone,
        invert_forward=invert_forward,
        next_episode_button=gamepad_next_button,
    )
