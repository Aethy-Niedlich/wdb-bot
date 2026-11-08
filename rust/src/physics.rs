//! Bit-perfect port of Bonk.io's Box2DWeb build.
//! Every function here must mirror the JS exactly — same order of operations,
//! same rounding (7 decimals via SafeTrig), same float width.

/// Bonk.io rounds all trig through this helper.
pub fn safe_sin(x: f64) -> f64 { (x.sin() * 1e7).round() / 1e7 }
pub fn safe_cos(x: f64) -> f64 { (x.cos() * 1e7).round() / 1e7 }

pub fn step(state: &mut GameState, inputs: &Inputs) {
    // ── exactly the same statements as the deobfuscated JS ──
    // 1. apply input accelerations
    // 2. integrate velocities
    // 3. resolve contacts (Box2DWeb solver order!)
    // 4. update abilities / cooldowns
    // 5. advance frame counter
    // ...
    state.t += 1;
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GameState {
    pub t: u64,
    // ... positions, velocities, charges, contacts
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Inputs {
    pub move_dir: i8,   // -1, 0, 1
    pub jump_dir: i8,   // -1, 0, 1
    pub heavy: bool,
}
