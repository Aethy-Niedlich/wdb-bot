const fs = require('fs');

function step(state, action) {
    let nextState = JSON.parse(JSON.stringify(state)); // deep copy
    const dt = 1.0 / 60.0;
    const gravity_x = 0.0;
    const gravity_y = -10.0;

    // Inputs
    let move_dir = action[0] - 1;
    let jump_dir = action[1] - 1;
    
    nextState.force_x = move_dir * 50.0;
    nextState.force_y = jump_dir * 50.0;

    // Box2D Integration
    nextState.vx += dt * (gravity_x + nextState.inv_mass * nextState.force_x);
    nextState.vy += dt * (gravity_y + nextState.inv_mass * nextState.force_y);
    
    // Damping
    let damping = Math.max(0.0, Math.min(1.0, 1.0 - dt * nextState.linear_damping));
    nextState.vx *= damping;
    nextState.vy *= damping;

    // Update position
    nextState.x += nextState.vx * dt;
    nextState.y += nextState.vy * dt;

    // Clear forces
    nextState.force_x = 0.0;
    nextState.force_y = 0.0;

    nextState.t += 1;
    return nextState;
}

function generateTraces(numFrames = 100) {
    let state = { 
        t: 0, x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, 
        force_x: 0.0, force_y: 0.0, 
        mass: 1.0, inv_mass: 1.0, linear_damping: 0.1 
    };
    let traces = [];

    for (let i = 0; i < numFrames; i++) {
        let action = [
            Math.floor(Math.random() * 3),
            Math.floor(Math.random() * 3),
            Math.floor(Math.random() * 2)
        ];

        let nextState = step(state, action);

        traces.push({
            seed: 0,
            state: state,
            action: action,
            next_state: nextState
        });

        state = nextState;
    }

    fs.writeFileSync('reference/frames.jsonl', traces.map(t => JSON.stringify(t)).join('\n') + '\n');
    console.log(`Generated ${numFrames} real JS traces to reference/frames.jsonl`);
}

generateTraces();