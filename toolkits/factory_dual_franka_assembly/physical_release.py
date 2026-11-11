"""Physical release qualification; never infer clearance from missing sensors."""
import math


def clear_contact_ready(state, *, step, openness, minimum_openness, forces, probes_valid, threshold, stable_steps):
    valid = (
        probes_valid
        and isinstance(openness, (int, float))
        and math.isfinite(openness)
        and openness >= minimum_openness
        and len(forces) == 2
        and all(isinstance(f, (int, float)) and math.isfinite(f) and 0 <= f < threshold for f in forces)
    )
    previous = state.get('step')
    if not valid:
        state.update(step=step, clear_steps=0)
        return False
    if previous != step:
        state['clear_steps'] = state.get('clear_steps', 0) + 1 if previous == step - 1 else 1
        state['step'] = step
    return state.get('clear_steps', 0) >= stable_steps
