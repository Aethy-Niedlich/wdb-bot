use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub t: u64,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub force_x: f64,
    pub force_y: f64,
    pub mass: f64,
    pub inv_mass: f64,
    pub linear_damping: f64,
}

impl GameState {
    pub fn reset(_seed: u64) -> Self {
        Self { 
            t: 0, x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, 
            force_x: 0.0, force_y: 0.0, 
            mass: 1.0, inv_mass: 1.0, 
            linear_damping: 0.1 
        }
    }
    
    pub fn reward(&self) -> f64 { 0.0 }
    pub fn is_done(&self) -> bool { self.t >= 100 }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Inputs {
    pub move_dir: i8,
    pub jump_dir: i8,
    pub heavy: bool,
}

// The real math starts here
pub fn step(state: &mut GameState, inputs: &Inputs) {
    let dt = 1.0 / 60.0;
    let gravity_x = 0.0;
    let gravity_y = -10.0; // Standard Box2D gravity

    // 1. Apply input forces (simulating what ApplyForce does)
    // In the real game, pressing 'D' sets m_force.x to a positive number.
    state.force_x = (inputs.move_dir as f64) * 50.0;
    state.force_y = (inputs.jump_dir as f64) * 50.0;

    // 2. Box2D Integration (Lines 5462-5467 from the screenshot)
    state.vx += dt * (gravity_x + state.inv_mass * state.force_x);
    state.vy += dt * (gravity_y + state.inv_mass * state.force_y);
    
    // Apply linear damping (air resistance)
    let damping = (1.0 - dt * state.linear_damping).clamp(0.0, 1.0);
    state.vx *= damping;
    state.vy *= damping;

    // 3. Update position (b2Sweep advance)
    state.x += state.vx * dt;
    state.y += state.vy * dt;

    // 4. Clear forces for the next frame
    state.force_x = 0.0;
    state.force_y = 0.0;

    state.t += 1;
}