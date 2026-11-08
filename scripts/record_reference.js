// scripts/record_reference.js
const fs = require('fs');

// Later, this will be: const { step, reset } = require('./bonk_physics.js');
// For now, we mock the JS engine to prove the harness works.

function mockStep(state, action) {
    // Exactly the same math as our Rust dummy engine
    let nextState = { ...state };
    let move = action[0] - 1;
    let jump = action[1] - 1;
    
    nextState.vx += move * 0.1;
    nextState.vy += jump * 0.1;
    nextState.x += nextState.vx;
    nextState.y += nextState.vy;
    nextState.t += 1;
    
    return nextState;
}

function generateTraces(numFrames = 100) {
    let state = { t: 0, x: 0.0, y: 0.0, vx: 0.0, vy: 0.0 };
    let traces = [];

    for (let i = 0; i < numFrames; i++) {
        let action = [
            Math.floor(Math.random() * 3),
            Math.floor(Math.random() * 3),
            Math.floor(Math.random() * 2)
        ];

        let nextState = mockStep(state, action);

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