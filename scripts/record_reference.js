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

     // Box2D Integration (Linear)
    nextState.vx += dt * (gravity_x + nextState.inv_mass * nextState.force_x);
    nextState.vy += dt * (gravity_y + nextState.inv_mass * nextState.force_y);
    
    // Box2D Integration (Angular)
    nextState.angular_velocity += dt * nextState.inv_inertia * nextState.torque;
    
    // Damping (Air resistance)
    let damp = Math.max(0.0, Math.min(1.0, 1.0 - dt * nextState.linear_damping));
    let ang_damp = Math.max(0.0, Math.min(1.0, 1.0 - dt * nextState.angular_damping));
    nextState.vx *= damp;
    nextState.vy *= damp;
    nextState.angular_velocity *= ang_damp;

    // Update position
    nextState.x += nextState.vx * dt;
    nextState.y += nextState.vy * dt;
    nextState.angle += nextState.angular_velocity * dt;
    
    // 4. SIMPLIFIED COLLISION (The Floor)
    const radius = 0.5;
    const floor_y = -5.0;
    
    if (nextState.y - radius < floor_y) {
        nextState.y = floor_y + radius;
        let dv_y = nextState.vy;
        let restitution = 0.2;
        let normal_mass = 1.0 / nextState.inv_mass;
        let vn = dv_y;
        
        if (vn < 0.0) {
            let lambda = -normal_mass * (1.0 + restitution) * vn;
            nextState.vy += nextState.inv_mass * lambda;
        }

        let friction = 0.3;
        let tangent_mass = 1.0 / nextState.inv_mass;
        let vt = nextState.vx;
        let max_friction = friction * Math.abs(normal_mass * -vn);
        let friction_impulse = -tangent_mass * vt;
        friction_impulse = Math.max(-max_friction, Math.min(max_friction, friction_impulse));
        nextState.vx += nextState.inv_mass * friction_impulse;
    }


    // Clear forces
    nextState.force_x = 0.0;
    nextState.force_y = 0.0;
    nextState.torque = 0.0;

    nextState.t += 1;
    return nextState;
}

function generateTraces(numFrames = 100) {
    // Notice the new fields added here
    let state = { 
        t: 0, x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, 
        angle: 0.0, angular_velocity: 0.0,
        force_x: 0.0, force_y: 0.0, torque: 0.0,
        mass: 1.0, inv_mass: 1.0, 
        inertia: 1.0, inv_inertia: 1.0,
        linear_damping: 0.1, angular_damping: 0.1
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