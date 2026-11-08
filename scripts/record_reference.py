import json
import random

def generate_mock_traces(num_frames=100):
    traces = []
    state = {"t": 0, "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}
    
    for i in range(num_frames):
        # 1. Generate a random action
        action = [random.randint(0, 2), random.randint(0, 2), random.randint(0, 1)]
        
        # 2. Apply the EXACT SAME math as the Rust engine (physics.rs)
        move_dir = action[0] - 1
        jump_dir = action[1] - 1
        
        next_state = state.copy()
        next_state["vx"] += move_dir * 0.1
        next_state["vy"] += jump_dir * 0.1
        next_state["x"] += next_state["vx"]
        next_state["y"] += next_state["vy"]
        next_state["t"] += 1
        
        traces.append({
            "seed": 0,
            "state": state,
            "action": action,
            "next_state": next_state
        })
        
        # Advance state for the next loop iteration
        state = next_state
        
    with open("reference/frames.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
            
if __name__ == "__main__":
    generate_mock_traces()
    print("Generated reference/frames.jsonl with matching physics")