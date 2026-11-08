import json
import random
import os

def generate_mock_replays(num_frames=5000):
    os.makedirs("reference", exist_ok=True)
    path = "reference/mock_replays.jsonl"
    
    with open(path, "w") as f:
        for _ in range(num_frames):
            # 5-float mock observation (matches vec_env.py)
            obs = [random.uniform(-1, 1) for _ in range(5)]
            
            # Mock human behavior: mostly move right (2), jump (2), heavy (0)
            move = 2 if random.random() > 0.2 else random.randint(0, 2)
            jump = 2 if random.random() > 0.5 else random.randint(0, 2)
            heavy = 0 if random.random() > 0.8 else 1
            
            f.write(json.dumps({"obs": obs, "action": [move, jump, heavy]}) + "\n")
            
    print(f"Generated {num_frames} frames of mock replay data at {path}")

if __name__ == "__main__":
    generate_mock_replays()