use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub t: u64,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
}

impl GameState {
    pub fn reset(_seed: u64) -> Self {
        Self { t: 0, x: 0.0, y: 0.0, vx: 0.0, vy: 0.0 }
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

pub fn safe_sin(x: f64) -> f64 { (x.sin() * 1e7).round() / 1e7 }
pub fn safe_cos(x: f64) -> f64 { (x.cos() * 1e7).round() / 1e7 }

pub fn step(state: &mut GameState, inputs: &Inputs) {
    // Dummy physics math for testing the bridge
    state.vx += (inputs.move_dir as f64) * 0.1;
    state.vy += (inputs.jump_dir as f64) * 0.1;
    state.x += state.vx;
    state.y += state.vy;
    state.t += 1;
}