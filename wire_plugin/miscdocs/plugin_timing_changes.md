# Summary of Changes for Plugin Timing

This document summarizes the changes made to enable timing of plugin `Compute` calls from the Python API.

## Design Overview

- **No changes to the core MuJoCo library.** Timing is implemented entirely in the elasticity plugin and Python.
- Plugins (Wire, Cable) already had internal timing variables (`total_compute_time_ms`, `compute_call_count`, `timing_enabled`). The new code exposes these via a C function and a Python wrapper.

---

## 1. Plugin-Side: `mj_getPluginTiming`

### New files

| File | Purpose |
|------|---------|
| `plugin/elasticity/plugin_timing.h` | Declares `mj_getPluginTiming` |
| `plugin/elasticity/plugin_timing.cc` | Implements `mj_getPluginTiming` |

### Behavior

- `mj_getPluginTiming(m, d, instance, &total_time_ms, &call_count)` reads timing data from `d->plugin_data[instance]` for Wire and Cable plugins.
- Identifies plugin type via `mjp_getPluginAtSlot(m->plugin[instance])->name` and casts to `Wire*` or `Cable*` to access `total_compute_time_ms` and `compute_call_count`.
- Returns `0` on success, `-1` on invalid parameters, `-2` if the plugin does not support timing.
- Exported as `extern "C" MJAPI` so it can be called from Python via `ctypes`.

### Build

- `plugin/elasticity/CMakeLists.txt`: Added `plugin_timing.cc` and `plugin_timing.h` to `MUJOCO_ELASTICITY_SRCS`.

---

## 2. Python-Side: `mj_step_timed`

### Modified files

| File | Change |
|------|--------|
| `python/mujoco/timing.py` | New module with `mj_step_timed` and ctypes-based access to `mj_getPluginTiming` |
| `python/mujoco/__init__.py` | Added `from mujoco.timing import mj_step_timed` |

### Behavior

- **`_load_timing_libraries()`**: Walks `mujoco/plugin/` for `.so`, `.dll`, `.dylib` files, loads each with `ctypes.CDLL`, and keeps handles that export `mj_getPluginTiming`.
- **`_get_plugin_timing_for_instance()`**: For a given `(model, data, instance)`, calls `mj_getPluginTiming` via ctypes with `model._address` and `data._address` as raw pointers.
- **`mj_step_timed(model, data, nstep, return_timing)`**: Runs `mj_step`; if `return_timing=True`, queries timing for all plugin instances and returns `{plugin_timing, total_plugin_time_ms, nsteps}`.

---

## 3. Existing Plugin Instrumentation (unchanged)

Wire and Cable already had:

- `total_compute_time_ms`, `compute_call_count` (per-instance)
- `timing_enabled` (configurable, e.g. via `timingEnabled` attribute)
- `std::chrono` timing around `Compute` when `timing_enabled` is true

No changes were made to `wire.cc` or `cable.cc` for this feature.

---

## Usage

```python
import mujoco

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

# Run steps and get plugin timing
timing = mujoco.mj_step_timed(model, data, nstep=100, return_timing=True)

for info in timing["plugin_timing"]:
    print(f"Instance {info['instance']}: {info['avg_time_ms']:.3f} ms/call, "
          f"{info['call_count']} calls")
print(f"Total plugin time: {timing['total_plugin_time_ms']:.3f} ms")
```

---

## Limitations

- Only Wire and Cable plugins are supported. Other plugins must add similar timing fields and be handled in `GetPluginTimingInternal` to be included.
- Timing is only recorded when `timingEnabled` (or equivalent) is true in the plugin config.
- Python discovers `mj_getPluginTiming` by loading plugin libraries under `mujoco/plugin/`. If the elasticity plugin is not built or not in that path, `mj_step_timed` will return empty timing.
