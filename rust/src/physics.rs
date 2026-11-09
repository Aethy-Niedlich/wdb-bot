use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Player {
    pub x: f64, pub y: f64, pub vx: f64, pub vy: f64,
    pub angle: f64, pub angular_velocity: f64,
    pub force_x: f64, pub force_y: f64, pub torque: f64,
    pub mass: f64, pub inv_mass: f64, pub inertia: f64, pub inv_inertia: f64,
    pub linear_damping: f64, pub angular_damping: f64,
    pub radius: f64, pub alive: bool,
    pub grapple_attached: bool, pub anchor_x: f64, pub anchor_y: f64, pub rope_length: f64,
}

impl Player {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x, y, vx: 0.0, vy: 0.0, angle: 0.0, angular_velocity: 0.0,
            force_x: 0.0, force_y: 0.0, torque: 0.0,
            mass: 1.0, inv_mass: 1.0, inertia: 1.0, inv_inertia: 1.0,
            linear_damping: 0.1, angular_damping: 0.1,
            radius: 0.5, alive: true,
            grapple_attached: false, anchor_x: 0.0, anchor_y: 0.0, rope_length: 0.0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Ball {
    pub x: f64, pub y: f64, pub vx: f64, pub vy: f64,
    pub radius: f64, pub exists: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub t: u64,
    pub p1: Player,
    pub p2: Player,
    pub ball: Ball,
    pub current_reward: f64,
    pub map_radius: f64, pub map_center_x: f64, pub map_center_y: f64,
    pub goal_x: f64, pub goal_y: f64, // NEW
    pub terminated: bool, pub truncated: bool,
}

impl GameState {
    pub fn reset(_seed: u64) -> Self {
        Self { 
            t: 0, 
            p1: Player::new(-2.0, 0.0),
            p2: Player::new(2.0, 0.0),
            ball: Ball { x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, radius: 0.4, exists: true },
            current_reward: 0.0,
            map_radius: 10.0, map_center_x: 0.0, map_center_y: -2.5,
            goal_x: 8.0, goal_y: -3.0, // Example goal position
            terminated: false, truncated: false,
        }
    }
    pub fn reward(&self) -> f64 { self.current_reward }
    pub fn is_done(&self) -> bool { self.terminated || self.truncated }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Inputs { pub move_dir: i8, pub jump_dir: i8, pub heavy: bool, pub grapple: bool }
pub struct Wall { pub x1: f64, pub y1: f64, pub x2: f64, pub y2: f64 }

fn calculate_phi_geo(p: &Player, cx: f64, cy: f64, r: f64) -> f64 {
    let w_c = 1.0; let w_v = 0.5;
    let dx = p.x - cx; let dy = p.y - cy;
    let d = (dx * dx + dy * dy).sqrt();
    let m = (r - d) / r;
    let ux = if d > 0.0 { -dx / d } else { 0.0 };
    let uy = if d > 0.0 { -dy / d } else { 0.0 };
    let v_mag = (p.vx * p.vx + p.vy * p.vy).sqrt();
    let v_hat_x = if v_mag > 0.0 { p.vx / v_mag } else { 0.0 };
    let v_hat_y = if v_mag > 0.0 { p.vy / v_mag } else { 0.0 };
    w_c * m + w_v * (v_hat_x * ux + v_hat_y * uy)
}

fn calculate_phi_h(p: &Player, ball: &Ball) -> f64 {
    if !ball.exists { return 0.0; }
    let w_h = 0.75; let d_0 = 3.0 * ball.radius;
    let h = 25.0; let k_steps = 10.0; // Claude's shorter horizon
    
    let dp_x = ball.x - p.x; let dp_y = ball.y - p.y;
    let dv_x = ball.vx - p.vx; let dv_y = ball.vy - p.vy;
    let mut g_min = f64::MAX; let mut t_star = 0.0;
    let is_grounded = p.y <= -4.5;
    
    for k in 0..=10 {
        let t_k = (k as f64) * (h / k_steps) / 60.0;
        let bx = ball.x + ball.vx * t_k;
        let by = ball.y + ball.vy * t_k - 0.5 * 10.0 * (t_k * t_k);
        let px = p.x + p.vx * t_k;
        let py = if is_grounded { p.y } else { p.y + p.vy * t_k - 0.5 * 10.0 * (t_k * t_k) };
        
        let gap = ((bx-px)*(bx-px) + (by-py)*(by-py)).sqrt() - ball.radius - p.radius;
        if gap < g_min { g_min = gap; t_star = t_k * 60.0 / h; }
    }
    let penalty = (1.0 - g_min / d_0).max(0.0).powi(2) * (1.0 - t_star);
    -w_h * penalty
}

fn calculate_phi_goal(ball: &Ball, goal_x: f64, goal_y: f64, l: f64) -> f64 {
    if !ball.exists { return 0.0; }
    let dx = ball.x - goal_x; let dy = ball.y - goal_y;
    -1.0 * (dx * dx + dy * dy).sqrt() / l
}

fn integrate_player(p: &mut Player, inputs: &Inputs, dt: f64) {
    p.force_x = (inputs.move_dir as f64) * 50.0;
    p.force_y = (inputs.jump_dir as f64) * 50.0;
    p.vx += dt * (p.inv_mass * p.force_x);
    p.vy += dt * (-10.0 + p.inv_mass * p.force_y);
    let damp = (1.0 - dt * p.linear_damping).clamp(0.0, 1.0);
    p.vx *= damp; p.vy *= damp;
    p.x += p.vx * dt; p.y += p.vy * dt;
    p.force_x = 0.0; p.force_y = 0.0;
}

fn integrate_ball(b: &mut Ball, dt: f64) {
    if !b.exists { return; }
    b.vy += dt * -10.0;
    b.x += b.vx * dt; b.y += b.vy * dt;
}

fn resolve_wall_collision(x: &mut f64, y: &mut f64, vx: &mut f64, vy: &mut f64, radius: f64, walls: &Vec<Wall>) {
    for wall in walls {
        let ab_x = wall.x2 - wall.x1; let ab_y = wall.y2 - wall.y1;
        let ap_x = *x - wall.x1; let ap_y = *y - wall.y1;
        let ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        let mut t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;
        t = t.clamp(0.0, 1.0);
        let cx = wall.x1 + t * ab_x; let cy = wall.y1 + t * ab_y;
        let dx = *x - cx; let dy = *y - cy;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < radius * radius {
            let dist = dist_sq.sqrt();
            let nx = if dist > 0.0 { dx / dist } else { 0.0 };
            let ny = if dist > 0.0 { dy / dist } else { 1.0 };
            *x += nx * (radius - dist); *y += ny * (radius - dist);
            let vn = *vx * nx + *vy * ny;
            if vn < 0.0 { *vx -= 1.2 * vn * nx; *vy -= 1.2 * vn * ny; }
        }
    }
}

fn get_closest_wall_point(p: &Player, walls: &Vec<Wall>) -> (f64, f64) {
    let mut best_dist_sq = f64::MAX; let mut best_x = 0.0; let mut best_y = 0.0;
    for wall in walls {
        let ab_x = wall.x2 - wall.x1; let ab_y = wall.y2 - wall.y1;
        let ap_x = p.x - wall.x1; let ap_y = p.y - wall.y1;
        let ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        let mut t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;
        t = t.clamp(0.0, 1.0);
        let cx = wall.x1 + t * ab_x; let cy = wall.y1 + t * ab_y;
        let dx = p.x - cx; let dy = p.y - cy;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < best_dist_sq { best_dist_sq = dist_sq; best_x = cx; best_y = cy; }
    }
    (best_x, best_y)
}

fn resolve_grapple(p: &mut Player, inputs: &Inputs, walls: &Vec<Wall>) {
    if inputs.grapple {
        if !p.grapple_attached {
            let (ax, ay) = get_closest_wall_point(p, walls);
            p.anchor_x = ax; p.anchor_y = ay;
            let dx = p.x - ax; let dy = p.y - ay;
            p.rope_length = (dx * dx + dy * dy).sqrt();
            p.grapple_attached = true;
        } else {
            let dx = p.x - p.anchor_x; let dy = p.y - p.anchor_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > p.rope_length {
                let nx = dx / dist; let ny = dy / dist;
                p.x = p.anchor_x + nx * p.rope_length; p.y = p.anchor_y + ny * p.rope_length;
                let vn = p.vx * nx + p.vy * ny;
                if vn > 0.0 { p.vx -= vn * nx; p.vy -= vn * ny; }
            }
        }
    } else { p.grapple_attached = false; }
}

pub fn step(state: &mut GameState, inputs1: &Inputs, inputs2: &Inputs) {
    let dt = 1.0 / 60.0;
    let l = state.map_radius;
    
    let phi_t = calculate_phi_geo(&state.p1, state.map_center_x, state.map_center_y, l) + 
                calculate_phi_h(&state.p1, &state.ball) + 
                calculate_phi_goal(&state.ball, state.goal_x, state.goal_y, l);

    integrate_player(&mut state.p1, inputs1, dt);
    integrate_player(&mut state.p2, inputs2, dt);
    integrate_ball(&mut state.ball, dt);

    let walls = vec![
        Wall { x1: -10.0, y1: -5.0, x2: 10.0, y2: -5.0 },
        Wall { x1: 5.0, y1: -5.0, x2: 8.0, y2: 0.0 },
        Wall { x1: -5.0, y1: -5.0, x2: -8.0, y2: 0.0 },
    ];
    
    resolve_wall_collision(&mut state.p1.x, &mut state.p1.y, &mut state.p1.vx, &mut state.p1.vy, state.p1.radius, &walls);
    resolve_wall_collision(&mut state.p2.x, &mut state.p2.y, &mut state.p2.vx, &mut state.p2.vy, state.p2.radius, &walls);
    resolve_wall_collision(&mut state.ball.x, &mut state.ball.y, &mut state.ball.vx, &mut state.ball.vy, state.ball.radius, &walls);
    
    resolve_grapple(&mut state.p1, inputs1, &walls);
    resolve_grapple(&mut state.p2, inputs2, &walls);

    // Check Goal
    if state.ball.exists {
        let gdx = state.ball.x - state.goal_x; let gdy = state.ball.y - state.goal_y;
        if (gdx*gdx + gdy*gdy).sqrt() < 1.0 {
            state.ball.exists = false;
            state.p2.alive = false; // Win for p1
        }
    }

    // Deadly Ball Collision
    if state.ball.exists {
        let bdx = state.ball.x - state.p1.x; let bdy = state.ball.y - state.p1.y;
        if (bdx*bdx + bdy*bdy).sqrt() < state.ball.radius + state.p1.radius { state.p1.alive = false; }
        
        let bdx2 = state.ball.x - state.p2.x; let bdy2 = state.ball.y - state.p2.y;
        if (bdx2*bdx2 + bdy2*bdy2).sqrt() < state.ball.radius + state.p2.radius { state.p2.alive = false; }
    }

    let kill_y = state.map_center_y - l;
    if state.p1.y < kill_y { state.p1.alive = false; }
    if state.p2.y < kill_y { state.p2.alive = false; }

    state.terminated = !state.p1.alive || !state.p2.alive;
    state.truncated = state.t >= 1000 && !state.terminated;

    let phi_next = if state.terminated { 0.0 } else { 
        calculate_phi_geo(&state.p1, state.map_center_x, state.map_center_y, l) + 
        calculate_phi_h(&state.p1, &state.ball) + 
        calculate_phi_goal(&state.ball, state.goal_x, state.goal_y, l)
    };
    
    let shaping = 0.99 * phi_next - phi_t;
    let terminal = if !state.p1.alive { -5.0 } else if !state.p2.alive { 10.0 } else { 0.0 };
    state.current_reward = 0.01 + shaping + terminal;

    state.t += 1;
}