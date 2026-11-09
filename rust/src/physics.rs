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
pub struct GameState {
    pub t: u64,
    pub p1: Player,
    pub p2: Player,
    pub current_reward: f64,
}

impl GameState {
    pub fn reset(_seed: u64) -> Self {
        Self { 
            t: 0, 
            p1: Player::new(-2.0, 0.0),
            p2: Player::new(2.0, 0.0),
            current_reward: 0.0,
        }
    }
    pub fn reward(&self) -> f64 { self.current_reward }
    pub fn is_done(&self) -> bool { !self.p1.alive || !self.p2.alive || self.t >= 1000 }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Inputs { pub move_dir: i8, pub jump_dir: i8, pub heavy: bool, pub grapple: bool }

pub struct Wall { pub x1: f64, pub y1: f64, pub x2: f64, pub y2: f64 }

fn calculate_phi(p: &Player) -> f64 {
    let r = 10.0; let w_c = 1.0; let w_v = 0.5;
    let d = (p.x * p.x + p.y * p.y).sqrt();
    let m = (r - d) / r;
    let ux = if d > 0.0 { -p.x / d } else { 0.0 };
    let uy = if d > 0.0 { -p.y / d } else { 0.0 };
    let v_mag = (p.vx * p.vx + p.vy * p.vy).sqrt();
    let v_hat_x = if v_mag > 0.0 { p.vx / v_mag } else { 0.0 };
    let v_hat_y = if v_mag > 0.0 { p.vy / v_mag } else { 0.0 };
    let dot = v_hat_x * ux + v_hat_y * uy;
    w_c * m + w_v * dot
}

fn integrate_player(p: &mut Player, inputs: &Inputs, dt: f64) {
    p.force_x = (inputs.move_dir as f64) * 50.0;
    p.force_y = (inputs.jump_dir as f64) * 50.0;

    p.vx += dt * (p.inv_mass * p.force_x);
    p.vy += dt * (-10.0 + p.inv_mass * p.force_y);
    p.angular_velocity += dt * p.inv_inertia * p.torque;
    
    let damp = (1.0 - dt * p.linear_damping).clamp(0.0, 1.0);
    let ang_damp = (1.0 - dt * p.angular_damping).clamp(0.0, 1.0);
    p.vx *= damp; p.vy *= damp; p.angular_velocity *= ang_damp;

    p.x += p.vx * dt; p.y += p.vy * dt; p.angle += p.angular_velocity * dt;

    p.force_x = 0.0; p.force_y = 0.0; p.torque = 0.0;
}

fn resolve_wall_collision(p: &mut Player, walls: &Vec<Wall>) {
    for wall in walls {
        let ab_x = wall.x2 - wall.x1; let ab_y = wall.y2 - wall.y1;
        let ap_x = p.x - wall.x1; let ap_y = p.y - wall.y1;
        let ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        let mut t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;
        t = t.clamp(0.0, 1.0);
        
        let closest_x = wall.x1 + t * ab_x;
        let closest_y = wall.y1 + t * ab_y;
        let dx = p.x - closest_x; let dy = p.y - closest_y;
        let dist_sq = dx * dx + dy * dy;
        
        if dist_sq < p.radius * p.radius {
            let dist = dist_sq.sqrt();
            let nx = if dist > 0.0 { dx / dist } else { 0.0 };
            let ny = if dist > 0.0 { dy / dist } else { 1.0 };
            
            let penetration = p.radius - dist;
            p.x += nx * penetration; p.y += ny * penetration;
            
            let vn = p.vx * nx + p.vy * ny;
            if vn < 0.0 {
                let lambda = -p.inv_mass * (1.0 + 0.2) * vn;
                p.vx += p.inv_mass * lambda * nx;
                p.vy += p.inv_mass * lambda * ny;
                
                let vt = p.vx * ny - p.vy * nx;
                let max_friction = 0.3 * lambda.abs();
                let mut friction_impulse = -vt / p.inv_mass;
                friction_impulse = friction_impulse.clamp(-max_friction, max_friction);
                p.vx += p.inv_mass * friction_impulse * ny;
                p.vy += p.inv_mass * friction_impulse * -nx;
            }
        }
    }
}

fn get_closest_wall_point(p: &Player, walls: &Vec<Wall>) -> (f64, f64) {
    let mut best_dist_sq = f64::MAX;
    let mut best_x = 0.0;
    let mut best_y = 0.0;
    
    for wall in walls {
        let ab_x = wall.x2 - wall.x1; let ab_y = wall.y2 - wall.y1;
        let ap_x = p.x - wall.x1; let ap_y = p.y - wall.y1;
        let ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        let mut t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;
        t = t.clamp(0.0, 1.0);
        
        let closest_x = wall.x1 + t * ab_x;
        let closest_y = wall.y1 + t * ab_y;
        let dx = p.x - closest_x; let dy = p.y - closest_y;
        let dist_sq = dx * dx + dy * dy;
        
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_x = closest_x;
            best_y = closest_y;
        }
    }
    (best_x, best_y)
}

fn resolve_grapple(p: &mut Player, inputs: &Inputs, walls: &Vec<Wall>) {
    if inputs.grapple {
        if !p.grapple_attached {
            let (ax, ay) = get_closest_wall_point(p, walls);
            p.anchor_x = ax;
            p.anchor_y = ay;
            let dx = p.x - ax;
            let dy = p.y - ay;
            p.rope_length = (dx * dx + dy * dy).sqrt();
            p.grapple_attached = true;
        } else {
            let dx = p.x - p.anchor_x;
            let dy = p.y - p.anchor_y;
            let dist = (dx * dx + dy * dy).sqrt();
            
            if dist > p.rope_length {
                let nx = dx / dist;
                let ny = dy / dist;
                
                p.x = p.anchor_x + nx * p.rope_length;
                p.y = p.anchor_y + ny * p.rope_length;
                
                let vn = p.vx * nx + p.vy * ny;
                if vn > 0.0 {
                    p.vx -= vn * nx;
                    p.vy -= vn * ny;
                }
            }
        }
    } else {
        p.grapple_attached = false;
    }
}

pub fn step(state: &mut GameState, inputs1: &Inputs, inputs2: &Inputs) {
    let dt = 1.0 / 60.0;
    let phi_t = calculate_phi(&state.p1);

    integrate_player(&mut state.p1, inputs1, dt);
    integrate_player(&mut state.p2, inputs2, dt);

    let walls = vec![
        Wall { x1: -10.0, y1: -5.0, x2: 10.0, y2: -5.0 },
        Wall { x1: 5.0, y1: -5.0, x2: 8.0, y2: 0.0 },
        Wall { x1: -5.0, y1: -5.0, x2: -8.0, y2: 0.0 },
    ];
    
    resolve_wall_collision(&mut state.p1, &walls);
    resolve_wall_collision(&mut state.p2, &walls);
    resolve_grapple(&mut state.p1, inputs1, &walls);
    resolve_grapple(&mut state.p2, inputs2, &walls);

    // Player vs Player Collision
    let dx = state.p2.x - state.p1.x;
    let dy = state.p2.y - state.p1.y;
    let dist_sq = dx * dx + dy * dy;
    let min_dist = state.p1.radius + state.p2.radius;

    if dist_sq < min_dist * min_dist && dist_sq > 0.0 {
        let dist = dist_sq.sqrt();
        let nx = dx / dist; let ny = dy / dist;

        let penetration = min_dist - dist;
        let p1_shift = penetration * (state.p1.inv_mass / (state.p1.inv_mass + state.p2.inv_mass));
        let p2_shift = penetration * (state.p2.inv_mass / (state.p1.inv_mass + state.p2.inv_mass));
        
        state.p1.x -= nx * p1_shift; state.p1.y -= ny * p1_shift;
        state.p2.x += nx * p2_shift; state.p2.y += ny * p2_shift;

        let rvx = state.p2.vx - state.p1.vx;
        let rvy = state.p2.vy - state.p1.vy;
        let vel_along_normal = rvx * nx + rvy * ny;

        if vel_along_normal < 0.0 {
            let e = 0.5; 
            let j = -(1.0 + e) * vel_along_normal / (state.p1.inv_mass + state.p2.inv_mass);
            let impulse_x = j * nx; let impulse_y = j * ny;
            state.p1.vx -= impulse_x * state.p1.inv_mass; state.p1.vy -= impulse_y * state.p1.inv_mass;
            state.p2.vx += impulse_x * state.p2.inv_mass; state.p2.vy += impulse_y * state.p2.inv_mass;
        }
    }

    // --- CLAUDE'S REWARD LOGIC ---
    if state.p1.y < -10.0 { state.p1.alive = false; }
    if state.p2.y < -10.0 { state.p2.alive = false; }

    let phi_next = calculate_phi(&state.p1);
    let shaping = 0.99 * phi_next - phi_t;
    
    let terminal = if !state.p1.alive { -5.0 } else if !state.p2.alive { 5.0 } else { 0.0 };
    state.current_reward = 0.01 + shaping + terminal;

    state.t += 1;
}