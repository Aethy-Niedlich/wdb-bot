const fs = require('fs');

function calculatePhi(p, cx, cy, r) {
    let w_c = 1.0; let w_v = 0.5;
    let dx = p.x - cx; let dy = p.y - cy;
    let d = Math.sqrt(dx * dx + dy * dy);
    let m = (r - d) / r;
    let ux = d > 0.0 ? -dx / d : 0.0;
    let uy = d > 0.0 ? -dy / d : 0.0;
    let v_mag = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
    let v_hat_x = v_mag > 0.0 ? p.vx / v_mag : 0.0;
    let v_hat_y = v_mag > 0.0 ? p.vy / v_mag : 0.0;
    let dot = v_hat_x * ux + v_hat_y * uy;
    return w_c * m + w_v * dot;
}

function integratePlayer(p, action, dt) {
    let move_dir = action[0] - 1; let jump_dir = action[1] - 1;
    p.force_x = move_dir * 50.0; p.force_y = jump_dir * 50.0;
    p.vx += dt * (p.inv_mass * p.force_x);
    p.vy += dt * (-10.0 + p.inv_mass * p.force_y);
    p.angular_velocity += dt * p.inv_inertia * p.torque;
    
    let damp = Math.max(0.0, Math.min(1.0, 1.0 - dt * p.linear_damping));
    let ang_damp = Math.max(0.0, Math.min(1.0, 1.0 - dt * p.angular_damping));
    p.vx *= damp; p.vy *= damp; p.angular_velocity *= ang_damp;

    p.x += p.vx * dt; p.y += p.vy * dt; p.angle += p.angular_velocity * dt;
    p.force_x = 0.0; p.force_y = 0.0; p.torque = 0.0;
}

function resolveWallCollision(p, walls) {
    for (let wall of walls) {
        let ab_x = wall.x2 - wall.x1; let ab_y = wall.y2 - wall.y1;
        let ap_x = p.x - wall.x1; let ap_y = p.y - wall.y1;
        let ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        let t = Math.max(0.0, Math.min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq));
        let closest_x = wall.x1 + t * ab_x; let closest_y = wall.y1 + t * ab_y;
        let dx = p.x - closest_x; let dy = p.y - closest_y;
        let dist_sq = dx * dx + dy * dy;
        
        if (dist_sq < p.radius * p.radius) {
            let dist = Math.sqrt(dist_sq);
            let nx = dist > 0.0 ? dx / dist : 0.0; let ny = dist > 0.0 ? dy / dist : 1.0;
            let penetration = p.radius - dist;
            p.x += nx * penetration; p.y += ny * penetration;
            
            let vn = p.vx * nx + p.vy * ny;
            if (vn < 0.0) {
                let lambda = -p.inv_mass * (1.0 + 0.2) * vn;
                p.vx += p.inv_mass * lambda * nx; p.vy += p.inv_mass * lambda * ny;
                let vt = p.vx * ny - p.vy * nx;
                let max_friction = 0.3 * Math.abs(lambda);
                let friction_impulse = Math.max(-max_friction, Math.min(max_friction, -vt / p.inv_mass));
                p.vx += p.inv_mass * friction_impulse * ny; p.vy += p.inv_mass * friction_impulse * -nx;
            }
        }
    }
}

function getClosestWallPoint(p, walls) {
    let best_dist_sq = Infinity; let best_x = 0; let best_y = 0;
    for (let wall of walls) {
        let ab_x = wall.x2 - wall.x1; let ab_y = wall.y2 - wall.y1;
        let ap_x = p.x - wall.x1; let ap_y = p.y - wall.y1;
        let ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        let t = Math.max(0.0, Math.min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq));
        let cx = wall.x1 + t * ab_x; let cy = wall.y1 + t * ab_y;
        let dx = p.x - cx; let dy = p.y - cy; let dist_sq = dx * dx + dy * dy;
        if (dist_sq < best_dist_sq) { best_dist_sq = dist_sq; best_x = cx; best_y = cy; }
    }
    return { x: best_x, y: best_y };
}

function resolveGrapple(p, action, walls) {
    let grapple = action[3] === 1;
    if (grapple) {
        if (!p.grapple_attached) {
            let anchor = getClosestWallPoint(p, walls);
            p.anchor_x = anchor.x; p.anchor_y = anchor.y;
            let dx = p.x - anchor.x; let dy = p.y - anchor.y;
            p.rope_length = Math.sqrt(dx * dx + dy * dy);
            p.grapple_attached = true;
        } else {
            let dx = p.x - p.anchor_x; let dy = p.y - p.anchor_y;
            let dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > p.rope_length) {
                let nx = dx / dist; let ny = dy / dist;
                p.x = p.anchor_x + nx * p.rope_length; p.y = p.anchor_y + ny * p.rope_length;
                let vn = p.vx * nx + p.vy * ny;
                if (vn > 0.0) { p.vx -= vn * nx; p.vy -= vn * ny; }
            }
        }
    } else { p.grapple_attached = false; }
}

function step(state, action1, action2) {
    let nextState = JSON.parse(JSON.stringify(state)); 
    const dt = 1.0 / 60.0;
    
    let phi_t = calculatePhi(nextState.p1, nextState.map_center_x, nextState.map_center_y, nextState.map_radius);

    integratePlayer(nextState.p1, action1, dt);
    integratePlayer(nextState.p2, action2, dt);

    const walls = [
        { x1: -10.0, y1: -5.0, x2: 10.0, y2: -5.0 },
        { x1: 5.0, y1: -5.0, x2: 8.0, y2: 0.0 },
        { x1: -5.0, y1: -5.0, x2: -8.0, y2: 0.0 }
    ];
    
    resolveWallCollision(nextState.p1, walls);
    resolveWallCollision(nextState.p2, walls);
    resolveGrapple(nextState.p1, action1, walls);
    resolveGrapple(nextState.p2, action2, walls);

    let dx = nextState.p2.x - nextState.p1.x; let dy = nextState.p2.y - nextState.p1.y;
    let dist_sq = dx * dx + dy * dy; let min_dist = nextState.p1.radius + nextState.p2.radius;

    if (dist_sq < min_dist * min_dist && dist_sq > 0.0) {
        let dist = Math.sqrt(dist_sq); let nx = dx / dist; let ny = dy / dist;
        let penetration = min_dist - dist;
        let p1_shift = penetration * (nextState.p1.inv_mass / (nextState.p1.inv_mass + nextState.p2.inv_mass));
        let p2_shift = penetration * (nextState.p2.inv_mass / (nextState.p1.inv_mass + nextState.p2.inv_mass));
        nextState.p1.x -= nx * p1_shift; nextState.p1.y -= ny * p1_shift;
        nextState.p2.x += nx * p2_shift; nextState.p2.y += ny * p2_shift;

        let rvx = nextState.p2.vx - nextState.p1.vx; let rvy = nextState.p2.vy - nextState.p1.vy;
        let vel_along_normal = rvx * nx + rvy * ny;

        if (vel_along_normal < 0.0) {
            let e = 0.5; let j = -(1.0 + e) * vel_along_normal / (nextState.p1.inv_mass + nextState.p2.inv_mass);
            let impulse_x = j * nx; let impulse_y = j * ny;
            nextState.p1.vx -= impulse_x * nextState.p1.inv_mass; nextState.p1.vy -= impulse_y * nextState.p1.inv_mass;
            nextState.p2.vx += impulse_x * nextState.p2.inv_mass; nextState.p2.vy += impulse_y * nextState.p2.inv_mass;
        }
    }

    let kill_y = nextState.map_center_y - nextState.map_radius;
    if (nextState.p1.y < kill_y) { nextState.p1.alive = false; }
    if (nextState.p2.y < kill_y) { nextState.p2.alive = false; }

    let phi_next = (!nextState.p1.alive || !nextState.p2.alive) ? 0.0 : calculatePhi(nextState.p1, nextState.map_center_x, nextState.map_center_y, nextState.map_radius);
    let shaping = 0.99 * phi_next - phi_t;
    
    let terminal = !nextState.p1.alive ? -5.0 : (!nextState.p2.alive ? 5.0 : 0.0);
    nextState.current_reward = 0.01 + shaping + terminal;

    nextState.t += 1;
    return nextState;
}

function generateTraces(numFrames = 100) {
    let state = { 
        t: 0, 
        map_radius: 10.0, map_center_x: 0.0, map_center_y: -2.5,
        current_reward: 0.0,
        p1: { x: -2.0, y: 0.0, vx: 0.0, vy: 0.0, angle: 0.0, angular_velocity: 0.0, force_x: 0.0, force_y: 0.0, torque: 0.0, mass: 1.0, inv_mass: 1.0, inertia: 1.0, inv_inertia: 1.0, linear_damping: 0.1, angular_damping: 0.1, radius: 0.5, alive: true, grapple_attached: false, anchor_x: 0.0, anchor_y: 0.0, rope_length: 0.0 },
        p2: { x: 2.0, y: 0.0, vx: 0.0, vy: 0.0, angle: 0.0, angular_velocity: 0.0, force_x: 0.0, force_y: 0.0, torque: 0.0, mass: 1.0, inv_mass: 1.0, inertia: 1.0, inv_inertia: 1.0, linear_damping: 0.1, angular_damping: 0.1, radius: 0.5, alive: true, grapple_attached: false, anchor_x: 0.0, anchor_y: 0.0, rope_length: 0.0 }
    };
    let traces = [];

    for (let i = 0; i < numFrames; i++) {
        let action1 = [Math.floor(Math.random() * 3), Math.floor(Math.random() * 3), Math.floor(Math.random() * 2), Math.floor(Math.random() * 2)];
        let action2 = [Math.floor(Math.random() * 3), Math.floor(Math.random() * 3), Math.floor(Math.random() * 2), Math.floor(Math.random() * 2)];

        let nextState = step(state, action1, action2);

        traces.push({ seed: 0, state: state, action1: action1, action2: action2, next_state: nextState });
        state = nextState;
    }
    fs.writeFileSync('reference/frames.jsonl', traces.map(t => JSON.stringify(t)).join('\n') + '\n');
    console.log(`Generated ${numFrames} real JS traces to reference/frames.jsonl`);
}

generateTraces();