"""
Minimal competence-improvement object/goal selector for multi-object QD grasp learning.

Each object has its own QD archive. After every completed QD generation on an object,
the caller reports only the new successful-archive size N and the duration of that
generation. The selector keeps all histories and computes the competence-improvement
(CI) signal internally; the caller never computes a CI.

    loop:
        idx = selector.select_goal()           # round-robin warm-up, then softmax
        N, dt = run_one_qd_generation(idx)     # external QD code
        selector.update(idx, archive_size=N, generation_duration_s=dt)

Supported CI signals (``selection_signal``), with window_size = w. The last 2w
observations of the object are split into a previous and a recent window:

    raw  = mean(N_recent) - mean(N_previous)                       [archive cells / generation]
    cov  = raw / n_cells[object]                                   [fraction of the grid / generation]
    rel  = raw / mean(N_previous)   (0.0 if mean(N_previous) == 0) [relative growth / generation]
    time = 60 * raw / (mean(T_recent) - mean(T_previous))          [archive cells / minute]

T is the cumulative learning time of that object only (sum of its own generation
durations), so time spent training other objects never enters CI_time. With w = 1
these reduce exactly to raw = N_now - N_prev, cov = raw / n_cells, rel = raw / N_prev
and time = 60 * raw / (duration of the last generation). A CI needs 2w observations.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence

import numpy as np

SELECTION_SIGNALS = ("raw", "cov", "rel", "time")

CI_UNITS = {
    "raw": "archive cells / generation",
    "cov": "fraction of archive grid / generation",
    "rel": "relative archive growth / generation",
    "time": "archive cells / minute",
}

# ---------------------------------------------------------------------------------------
# Experiment-specific temperature presets. NOT universal values.
#
# These temperatures were calibrated on the recorded QD runs of the current experiment
# (ycb mug / banana / power drill / spatula, runs 0-3, 5-8, 10-13, 15-18): each one makes
# the mean over replica sets of the best object's probability, right after a 2-generation
# warm-up with window_size = 1, equal to the target p_max. They depend on the scale of the
# CI values in those data, so they must be recalibrated for other objects, grids or QD
# settings. No preset exists for "time": it has not been calibrated yet.
# ---------------------------------------------------------------------------------------
CALIBRATED_TEMPERATURES: dict[float, dict[str, float]] = {
    0.60: {"raw": 63.3803, "cov": 0.00152634, "rel": 0.0485533},
    0.80: {"raw": 36.2592, "cov": 0.000896504, "rel": 0.0228172},
}


def experimental_temperature(signal: str, target_pmax: float = 0.60) -> float:
    """Temperature calibrated on the current experiment's data (see CALIBRATED_TEMPERATURES).

    Convenience only: the selector itself always takes an explicit ``temperature``.
    """
    if target_pmax not in CALIBRATED_TEMPERATURES:
        raise KeyError(f"no calibrated preset for target p_max = {target_pmax}; "
                       f"available: {sorted(CALIBRATED_TEMPERATURES)}")
    presets = CALIBRATED_TEMPERATURES[target_pmax]
    if signal not in presets:
        raise KeyError(f"no calibrated preset for signal {signal!r} at target p_max = "
                       f"{target_pmax}; available: {sorted(presets)}. "
                       f"'time' has not been calibrated yet: pass an explicit temperature.")
    return presets[signal]


class GoalSelector:
    """Selects which object gets the next QD generation.

    Parameters
    ----------
    objects:
        Object identifiers. Objects are addressed by their index in this sequence.
    temperature:
        Softmax temperature, in the units of the selection signal (see CI_UNITS).
    selection_signal:
        "raw" (default), "cov", "rel" or "time".
    n_cells:
        Total number of archive cells of each object, aligned with ``objects``.
        Required for selection_signal="cov"; optional otherwise (if given, CI_cov is
        also recorded for analysis).
    warmup_generations:
        Generations every object receives, round-robin, before softmax selection.
        Must be >= 2 * window_size so every object has a CI when selection starts.
    window_size:
        Size w of each of the two compared windows.
    extrinsic_bias_enabled:
        Hook for a future extrinsic bias. Must be False in this version.
    seed:
        Optional seed for the internal numpy Generator (reproducible tests).
    """

    def __init__(
        self,
        objects: Sequence[Hashable],
        temperature: float,
        selection_signal: str = "raw",
        n_cells: Sequence[float] | None = None,
        warmup_generations: int = 2,
        window_size: int = 1,
        extrinsic_bias_enabled: bool = False,
        seed: int | None = None,
    ) -> None:
        if len(objects) == 0:
            raise ValueError("objects must not be empty")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be a finite value > 0, got {temperature}")
        if selection_signal not in SELECTION_SIGNALS:
            raise ValueError(f"selection_signal must be one of {SELECTION_SIGNALS}, "
                             f"got {selection_signal!r}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if warmup_generations < 2 * window_size:
            raise ValueError(
                f"warmup_generations ({warmup_generations}) must be >= 2 * window_size "
                f"({2 * window_size}): a CI compares two windows of window_size observations")
        if extrinsic_bias_enabled:
            raise NotImplementedError("extrinsic bias is a future extension; "
                                      "keep extrinsic_bias_enabled=False")

        self.objects = list(objects)
        self.n_objects = len(self.objects)
        self.n_cells = self._validate_n_cells(n_cells)
        if selection_signal == "cov" and self.n_cells is None:
            raise ValueError('selection_signal="cov" requires n_cells (one value per object)')

        self.temperature = float(temperature)
        self.selection_signal = selection_signal
        self.warmup_generations = warmup_generations
        self.window_size = window_size
        self.extrinsic_bias_enabled = extrinsic_bias_enabled
        self.rng = np.random.default_rng(seed)

        # Signals that can be computed with the information available ("cov" needs n_cells).
        self.recorded_signals = tuple(s for s in SELECTION_SIGNALS
                                      if s != "cov" or self.n_cells is not None)

        # Per-object raw histories (lists indexed by object index). Append-only, never smoothed.
        self.archive_size_history: list[list[float]] = [[] for _ in self.objects]
        self.generation_duration_history: list[list[float]] = [[] for _ in self.objects]
        self.elapsed_time_history: list[list[float]] = [[] for _ in self.objects]  # cumulative [s]

        # CI of every recorded signal, per object; the selection uses self.selection_signal.
        self.ci_histories: dict[str, list[list[float]]] = {
            s: [[] for _ in self.objects] for s in self.recorded_signals}
        self.latest_cis: dict[str, list[float | None]] = {
            s: [None] * self.n_objects for s in self.recorded_signals}

        # Global decision log; entry k refers to the k-th select_goal() call.
        # score/probability entries are None for warm-up (round-robin) picks.
        self.selection_history: list[int] = []
        self.score_history: list[np.ndarray | None] = []
        self.probability_history: list[np.ndarray | None] = []
        self.last_scores: np.ndarray | None = None
        self.last_probabilities: np.ndarray | None = None

    # ------------------------------------------------------------------ update

    def update(self, object_idx: int, archive_size: float, generation_duration_s: float) -> None:
        """Record one completed QD generation on ``object_idx`` and update its CI.

        ``generation_duration_s`` is the duration (seconds) of that generation only; the
        selector accumulates it into the object's own learning time.
        """
        self._check_idx(object_idx)
        if not (np.isfinite(archive_size) and np.isfinite(generation_duration_s)):
            raise ValueError(f"non-finite input: archive_size={archive_size}, "
                             f"generation_duration_s={generation_duration_s}")
        if generation_duration_s <= 0:
            raise ValueError(f"generation_duration_s must be > 0, got {generation_duration_s}")

        sizes = self.archive_size_history[object_idx]
        times = self.elapsed_time_history[object_idx]
        sizes.append(float(archive_size))
        self.generation_duration_history[object_idx].append(float(generation_duration_s))
        times.append((times[-1] if times else 0.0) + float(generation_duration_s))

        if len(sizes) >= 2 * self.window_size:
            for signal, value in self._compute_cis(object_idx).items():
                self.ci_histories[signal][object_idx].append(value)
                self.latest_cis[signal][object_idx] = value

    def _compute_cis(self, object_idx: int) -> dict[str, float]:
        """All recorded CI signals from the last 2w observations of the object."""
        w = self.window_size
        sizes = self.archive_size_history[object_idx]
        times = self.elapsed_time_history[object_idx]
        n_prev, n_recent = float(np.mean(sizes[-2 * w:-w])), float(np.mean(sizes[-w:]))
        t_prev, t_recent = float(np.mean(times[-2 * w:-w])), float(np.mean(times[-w:]))

        raw = n_recent - n_prev
        cis = {
            "raw": raw,
            "rel": raw / n_prev if n_prev != 0 else 0.0,
            "time": 60.0 * raw / (t_recent - t_prev),  # t_recent > t_prev: durations > 0
        }
        if self.n_cells is not None:
            cis["cov"] = raw / self.n_cells[object_idx]
        return {s: cis[s] for s in self.recorded_signals}

    # --------------------------------------------------------------- selection

    def select_goal(self) -> int:
        """Return the index of the object that should get the next QD generation."""
        warmup_idx = self._next_warmup_object()
        if warmup_idx is not None:
            self._log(warmup_idx, None, None)
            return warmup_idx

        scores = np.array([self._compute_score(i) for i in range(self.n_objects)], dtype=float)
        probabilities = self._softmax(scores)
        selected = int(self.rng.choice(self.n_objects, p=probabilities))
        self._log(selected, scores, probabilities)
        return selected

    def in_warmup(self) -> bool:
        return self._next_warmup_object() is not None

    def get_probabilities(self) -> np.ndarray:
        """Current selection distribution without sampling or logging (for debugging)."""
        if self.in_warmup():
            raise RuntimeError("probabilities are undefined during warm-up")
        return self._softmax(np.array([self._compute_score(i) for i in range(self.n_objects)],
                                      dtype=float))

    # ------------------------------------------------------- per-object views

    @property
    def ci_history(self) -> list[list[float]]:
        """CI history of the selection signal, per object."""
        return self.ci_histories[self.selection_signal]

    @property
    def latest_ci(self) -> list[float | None]:
        """Latest CI of the selection signal, per object (None before 2w observations)."""
        return self.latest_cis[self.selection_signal]

    def n_generations(self, object_idx: int) -> int:
        self._check_idx(object_idx)
        return len(self.archive_size_history[object_idx])

    def learning_time(self, object_idx: int) -> float:
        """Cumulative learning time [s] spent on ``object_idx``."""
        self._check_idx(object_idx)
        times = self.elapsed_time_history[object_idx]
        return times[-1] if times else 0.0

    def selection_steps(self, object_idx: int, post_warmup_only: bool = False) -> list[int]:
        """Positions in selection_history at which ``object_idx`` was selected."""
        self._check_idx(object_idx)
        return [k for k, s in enumerate(self.selection_history)
                if s == object_idx and not (post_warmup_only and self.score_history[k] is None)]

    def summary(self) -> list[dict[str, Any]]:
        """Per-object overview. Warm-up picks are round-robin, not softmax decisions,
        so they are counted separately from the post-warm-up selections."""
        rows = []
        for i in range(self.n_objects):
            total = len(self.selection_steps(i))
            post = len(self.selection_steps(i, post_warmup_only=True))
            rows.append({
                "object": self.objects[i],
                "generations": self.n_generations(i),
                "archive_size": (self.archive_size_history[i][-1]
                                 if self.archive_size_history[i] else None),
                "learning_time_s": self.learning_time(i),
                "selection_signal": self.selection_signal,
                "latest_ci": self.latest_ci[i],
                "total_times_selected": total,
                "warmup_times_selected": total - post,
                "post_warmup_times_selected": post,
            })
        return rows

    # --------------------------------------------------------------- internals

    def _compute_score(self, object_idx: int) -> float:
        ci = self.latest_ci[object_idx]
        if ci is None:
            raise RuntimeError(f"object {object_idx} has no CI yet; "
                               f"was update() called after every generation?")
        if not self.extrinsic_bias_enabled:
            return ci
        # Future hook: e.g. return ci + self._extrinsic_bias(object_idx)
        raise NotImplementedError

    def _next_warmup_object(self) -> int | None:
        """Round-robin: lowest-index object with the fewest generations, while below warm-up."""
        counts = [len(h) for h in self.archive_size_history]
        if min(counts) >= self.warmup_generations:
            return None
        return int(np.argmin(counts))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # Same form as the old selector; subtracting the max prevents overflow.
        # Equal scores -> exp(0) everywhere -> uniform distribution.
        e_x = np.exp((x - np.max(x)) / self.temperature)
        return e_x / e_x.sum()

    def _log(self, selected: int, scores: np.ndarray | None,
             probabilities: np.ndarray | None) -> None:
        self.selection_history.append(selected)
        self.score_history.append(scores)
        self.probability_history.append(probabilities)
        if scores is not None:
            self.last_scores = scores
            self.last_probabilities = probabilities

    def _validate_n_cells(self, n_cells: Sequence[float] | None) -> list[float] | None:
        if n_cells is None:
            return None
        values = [float(v) for v in n_cells]
        if len(values) != self.n_objects:
            raise ValueError(f"n_cells must have one value per object "
                             f"({self.n_objects}), got {len(values)}")
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError(f"n_cells must be finite and > 0, got {values}")
        return values

    def _check_idx(self, object_idx: int) -> None:
        if not 0 <= object_idx < self.n_objects:
            raise IndexError(f"object_idx {object_idx} out of range [0, {self.n_objects})")