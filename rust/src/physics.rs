use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub t: u64,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub angle: f64,         // NEW: Rotation
    pub angular_velocity: f64, // NEW: Spin
    pub force_x: f64,
    pub force_y: f64,
    pub torque: f64,        // NEW: Rotational force
    pub mass: f64,
    pub inv_mass: f64,
    pub inertia: f64,       // NEW: Moment of inertia
    pub inv_inertia: f64,   // NEW: Inverse moment of inertia
    pub linear_damping: f64,
    pub angular_damping: f64, // NEW
}

impl GameState {
    pub fn reset(_seed: u64) -> Self {
        Self { 
            t: 0, x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, 
            angle: 0.0, angular_velocity: 0.0,
            force_x: 0.0, force_y: 0.0, torque: 0.0,
            mass: 1.0, inv_mass: 1.0, 
            inertia: 1.0, inv_inertia: 1.0,
            linear_damping: 0.1, angular_damping: 0.1
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

pub fn step(state: &mut GameState, inputs: &Inputs) {
    let dt = 1.0 / 60.0;
    let gravity_x = 0.0;
    let gravity_y = -10.0; 

    // 1. Apply Input Forces
    state.force_x = (inputs.move_dir as f64) * 50.0;
    state.force_y = (inputs.jump_dir as f64) * 50.0;

    // 2. Box2D Integration
    state.vx += dt * (gravity_x + state.inv_mass * state.force_x);
    state.vy += dt * (gravity_y + state.inv_mass * state.force_y);
    state.angular_velocity += dt * state.inv_inertia * state.torque;
    
    let damp = (1.0 - dt * state.linear_damping).clamp(0.0, 1.0);
    let ang_damp = (1.0 - dt * state.angular_damping).clamp(0.0, 1.0);
    state.vx *= damp;
    state.vy *= damp;
    state.angular_velocity *= ang_damp;

    // 3. Update Position
    state.x += state.vx * dt;
    state.y += state.vy * dt;
    state.angle += state.angular_velocity * dt;

    // 4. SIMPLIFIED COLLISION (The Floor)
    // Let's pretend the player is a circle with radius 0.5, and the floor is at y = -5.0
    let radius = 0.5;
    let floor_y = -5.0;
    
    if state.y - radius < floor_y {
        // We hit the floor! Push the player back up so they don't sink
        state.y = floor_y + radius;
        
        // Calculate relative velocity (moving down)
        let dv_y = state.vy;
        
        // Normal is (0, 1) pointing up. Tangent is (1, 0) pointing right.
        // Restitution (bounciness) = 0.2 (Bonk players don't bounce much)
        let restitution = 0.2;
        
        // Box2D Impulse Math for the normal (vertical) force
        let normal_mass = 1.0 / (state.inv_mass); // Floor has 0 invMass
        let vn = dv_y; // Normal velocity
        
        if vn < 0.0 {
            let lambda = -normal_mass * (1.0 + restitution) * vn;
            // Apply the impulse to the velocity
            state.vy += state.inv_mass * lambda;
        }

        // Box2D Impulse Math for friction (horizontal)
        let friction = 0.3;
        let tangent_mass = 1.0 / (state.inv_mass);
        let vt = state.vx;
        let max_friction = friction * (normal_mass * -vn).abs();
        let mut friction_impulse = -tangent_mass * vt;
        
        // Clamp friction
        friction_impulse = friction_impulse.clamp(-max_friction, max_friction);
        
        // Apply friction to horizontal velocity
        state.vx += state.inv_mass * friction_impulse;
    }

    // Clear forces
    state.force_x = 0.0;
    state.force_y = 0.0;
    state.torque = 0.0;
    state.t += 1;
}