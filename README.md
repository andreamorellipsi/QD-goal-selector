# QD goal selector

`qd_goal_selector.py` implements a goal selector for multi-object QD grasp learning.

Each object has its own QD archive. The selector is called **between QD generations** and decides which object should receive the next generation of learning.

The QD algorithm itself is not modified by this class. The external learning loop only needs to:

1. ask the selector which object should be trained next;
2. run one QD generation for that object;
3. report the updated successful-archive size and the duration of that generation.

The selector stores the history of each object and computes the learning-progress signal internally.

---

## Basic usage

```python
from qd_goal_selector import GoalSelector

objects = ["mug", "banana", "drill", "spatula"]

selector = GoalSelector(
    objects=objects,
    temperature=0.05,
    selection_signal="rel",
    n_cells=[35937, 50653, 64000, 140608],
    warmup_generations=2,
    window_size=1,
    seed=0,
)

while learning_budget_available:

    # Select the object that should receive the next QD generation
    object_idx = selector.select_goal()

    # External QD code
    archive_size, generation_duration_s = run_one_qd_generation(
        objects[object_idx]
    )

    # Update the selector after the generation has finished
    selector.update(
        object_idx,
        archive_size=archive_size,
        generation_duration_s=generation_duration_s,
    )
```

`select_goal()` returns the **index** of the selected object in the `objects` sequence.

`update()` must be called after every completed QD generation.

The value passed as `archive_size` is the **current total number of occupied successful cells** in the archive, not only the number of new cells added during the last generation.

The value passed as `generation_duration_s` is the duration, in seconds, of that generation only. The selector accumulates the learning time internally for each object.

---

## Constructor parameters

### `objects`

Sequence containing the identifiers of the available objects.

```python
objects = ["mug", "banana", "drill", "spatula"]
```

The selector internally works with their indices. For example, if `select_goal()` returns `2`, the selected object is `objects[2]`.

---

### `selection_signal`

Defines how learning progress, called competence improvement (`CI`), is measured.

Four signals are currently available:

- `"raw"`: absolute increase in successful archive size;
- `"cov"`: archive increase normalized by the total number of archive cells;
- `"rel"`: archive increase relative to the previous archive size;
- `"time"`: archive increase per minute of learning time.

With the default `window_size=1`, the signals are:

```text
CI_raw  = N_now - N_prev

CI_cov  = (N_now - N_prev) / n_cells

CI_rel  = (N_now - N_prev) / N_prev

CI_time = 60 * (N_now - N_prev) / generation_duration_s
```

If `N_prev = 0`, `CI_rel` is set to `0`.

The selected CI is used as the score given to the softmax.

---

### `temperature`

Softmax temperature used to convert the current CI scores into object-selection probabilities.

A **lower temperature** makes the selector more selective: objects with larger CI values receive more probability.

A **higher temperature** makes the probability distribution more uniform and therefore increases exploration.

The numerical scale of the temperature depends on the selected CI signal. A temperature value that works for one signal should not be assumed to work for another signal.

---

### `n_cells`

Total number of archive cells for each object, in the same order as `objects`.

Example:

```python
n_cells = [
    35937,   # mug
    50653,   # banana
    64000,   # drill
    140608,  # spatula
]
```

This parameter is required when using:

```python
selection_signal="cov"
```

For the other signals it is optional.

If `n_cells` is provided while another signal is used for selection, `CI_cov` is still computed and stored internally for analysis.

---

### `warmup_generations`

Number of generations initially assigned to every object before softmax-based selection starts.

For example:

```python
warmup_generations=2
```

with four objects gives an initial round-robin sequence equivalent to:

```text
object 0
object 1
object 2
object 3
object 0
object 1
object 2
object 3
```

The warm-up ensures that enough observations are available to compute a CI value for every object before probabilistic selection begins.

It must satisfy:

```text
warmup_generations >= 2 * window_size
```

---

### `window_size`

Controls how many observations are used to compute CI.

With:

```python
window_size=1
```

the selector directly compares two consecutive generations.

With a larger value, the last `2 * window_size` observations are divided into two groups:

```text
previous window
recent window
```

The CI is then computed from the difference between the **mean archive size** of the recent window and the mean archive size of the previous window.

Therefore:

- `window_size=1` reacts directly to changes between consecutive generations;
- larger values smooth short-term variations and measure progress across two successive windows.

A CI value requires at least `2 * window_size` observations for that object.

---

### `seed`

Optional random seed for the internal NumPy random generator.

```python
seed=0
```

Using a fixed seed is useful for reproducible experiments and tests.

After the warm-up, the selector samples an object according to the softmax probabilities. Therefore, different seeds can produce different selection sequences even when the CI values are identical.

---

### `extrinsic_bias_enabled`

Reserved for a future external bias, for example user or designer feedback.

This functionality is not implemented yet and must currently remain:

```python
extrinsic_bias_enabled=False
```

---

## Temperature presets used in the current experiment

The module includes the helper:

```python
experimental_temperature(signal, target_pmax)
```

Example:

```python
from qd_goal_selector import experimental_temperature

T = experimental_temperature("rel", 0.60)
```

The available target values are currently:

```text
0.60
0.80
```

These temperatures were calibrated specifically on the recorded mug, banana, power-drill and spatula experiments used in the current analysis.

They are provided only as a convenience for reproducing the current experiment and are **not universal values**.

They should be recalibrated when changing:

- objects;
- archive-grid sizes;
- QD settings;
- the scale or definition of the CI signal.

No calibrated preset is currently provided for:

```python
selection_signal="time"
```

For the time-based signal, an explicit temperature must be provided.

---

## Inspecting the selector

The selector stores the learning and selection history internally.

Some useful attributes and methods are listed below.

### Latest CI

```python
selector.latest_ci
```

Returns the latest CI value for each object for the signal currently used for selection.

Before enough observations are available, the corresponding value is `None`.

---

### CI history

```python
selector.ci_history
```

Returns the complete CI history for each object for the signal currently used for selection.

The selector also stores the histories of the other CI signals that can be computed from the available information.

---

### Current softmax probabilities

```python
selector.get_probabilities()
```

Returns the current object-selection probability distribution **without sampling a new object and without adding an entry to the selection history**.

This method is only available after the warm-up.

---

### Number of completed generations

```python
selector.n_generations(object_idx)
```

Returns the number of QD generations completed for the specified object.

---

### Learning time

```python
selector.learning_time(object_idx)
```

Returns the cumulative learning time, in seconds, spent on the specified object.

Only the durations of generations executed for that object are included.

---

### Summary

```python
selector.summary()
```

Returns a per-object summary containing:

- number of completed generations;
- current archive size;
- cumulative learning time;
- current selection signal;
- latest CI;
- total number of times the object was selected;
- number of warm-up selections;
- number of post-warm-up softmax selections.

Warm-up selections are reported separately because they are deterministic round-robin assignments rather than softmax decisions.

---

## Typical control loop

The intended integration pattern is:

```text
select object
    ↓
run one QD generation
    ↓
obtain the new successful archive size N
    ↓
measure the duration of that generation
    ↓
update GoalSelector
    ↓
select the next object
```

The Goal Selector only decides how the available learning budget is distributed across objects.

The QD archives, the execution of each QD generation, and the stopping condition of the overall learning process remain managed by the external learning pipeline.
