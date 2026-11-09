pub mod physics;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::physics::{GameState, Inputs, step};

#[pyclass]
struct RustEnv {
    state: GameState,
}

#[pymethods]
impl RustEnv {
    #[new]
    fn new(seed: u64) -> Self {
        Self { state: GameState::reset(seed) }
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) -> PyResult<PyObject> {
        self.state = GameState::reset(seed.unwrap_or(0));
        Python::with_gil(|py| Ok(self.state.to_py(py)))
    }

    fn step(&mut self, action1: Vec<i8>, action2: Vec<i8>) -> PyResult<(PyObject, f64, bool)> {
        let inputs1 = Inputs {
            move_dir: action1[0] - 1, jump_dir: action1[1] - 1,
            heavy: action1[2] == 1, grapple: action1[3] == 1,
        };
        let inputs2 = Inputs {
            move_dir: action2[0] - 1, jump_dir: action2[1] - 1,
            heavy: action2[2] == 1, grapple: action2[3] == 1,
        };
        
        step(&mut self.state, &inputs1, &inputs2);
        
        let reward = self.state.reward();
        let done = self.state.is_done();
        Python::with_gil(|py| Ok((self.state.to_py(py), reward, done)))
    }

    fn step_json(&mut self, state_json: &str, action1: Vec<i8>, action2: Vec<i8>) -> PyResult<String> {
        let mut s: GameState = serde_json::from_str(state_json).unwrap();
        let inputs1 = Inputs {
            move_dir: action1[0] - 1, jump_dir: action1[1] - 1,
            heavy: action1[2] == 1, grapple: action1[3] == 1,
        };
        let inputs2 = Inputs {
            move_dir: action2[0] - 1, jump_dir: action2[1] - 1,
            heavy: action2[2] == 1, grapple: action2[3] == 1,
        };
        
        step(&mut s, &inputs1, &inputs2);
        Ok(serde_json::to_string(&s).unwrap())
    }
}

impl GameState {
    fn to_py(&self, py: Python) -> PyObject {
        let d = PyDict::new_bound(py); 
        d.set_item("t", self.t).unwrap();
        d.set_item("current_reward", self.current_reward).unwrap();
        d.set_item("map_radius", self.map_radius).unwrap();
        d.set_item("map_center_x", self.map_center_x).unwrap();
        d.set_item("map_center_y", self.map_center_y).unwrap();
        
        let p1_dict = PyDict::new_bound(py);
        p1_dict.set_item("x", self.p1.x).unwrap();
        p1_dict.set_item("y", self.p1.y).unwrap();
        p1_dict.set_item("vx", self.p1.vx).unwrap();
        p1_dict.set_item("vy", self.p1.vy).unwrap();
        p1_dict.set_item("angle", self.p1.angle).unwrap();
        p1_dict.set_item("angular_velocity", self.p1.angular_velocity).unwrap();
        p1_dict.set_item("force_x", self.p1.force_x).unwrap();
        p1_dict.set_item("force_y", self.p1.force_y).unwrap();
        p1_dict.set_item("torque", self.p1.torque).unwrap();
        p1_dict.set_item("mass", self.p1.mass).unwrap();
        p1_dict.set_item("inv_mass", self.p1.inv_mass).unwrap();
        p1_dict.set_item("inertia", self.p1.inertia).unwrap();
        p1_dict.set_item("inv_inertia", self.p1.inv_inertia).unwrap();
        p1_dict.set_item("linear_damping", self.p1.linear_damping).unwrap();
        p1_dict.set_item("angular_damping", self.p1.angular_damping).unwrap();
        p1_dict.set_item("radius", self.p1.radius).unwrap();
        p1_dict.set_item("alive", self.p1.alive).unwrap();
        p1_dict.set_item("grapple_attached", self.p1.grapple_attached).unwrap();
        p1_dict.set_item("anchor_x", self.p1.anchor_x).unwrap();
        p1_dict.set_item("anchor_y", self.p1.anchor_y).unwrap();
        p1_dict.set_item("rope_length", self.p1.rope_length).unwrap();

        let p2_dict = PyDict::new_bound(py);
        p2_dict.set_item("x", self.p2.x).unwrap();
        p2_dict.set_item("y", self.p2.y).unwrap();
        p2_dict.set_item("vx", self.p2.vx).unwrap();
        p2_dict.set_item("vy", self.p2.vy).unwrap();
        p2_dict.set_item("angle", self.p2.angle).unwrap();
        p2_dict.set_item("angular_velocity", self.p2.angular_velocity).unwrap();
        p2_dict.set_item("force_x", self.p2.force_x).unwrap();
        p2_dict.set_item("force_y", self.p2.force_y).unwrap();
        p2_dict.set_item("torque", self.p2.torque).unwrap();
        p2_dict.set_item("mass", self.p2.mass).unwrap();
        p2_dict.set_item("inv_mass", self.p2.inv_mass).unwrap();
        p2_dict.set_item("inertia", self.p2.inertia).unwrap();
        p2_dict.set_item("inv_inertia", self.p2.inv_inertia).unwrap();
        p2_dict.set_item("linear_damping", self.p2.linear_damping).unwrap();
        p2_dict.set_item("angular_damping", self.p2.angular_damping).unwrap();
        p2_dict.set_item("radius", self.p2.radius).unwrap();
        p2_dict.set_item("alive", self.p2.alive).unwrap();
        p2_dict.set_item("grapple_attached", self.p2.grapple_attached).unwrap();
        p2_dict.set_item("anchor_x", self.p2.anchor_x).unwrap();
        p2_dict.set_item("anchor_y", self.p2.anchor_y).unwrap();
        p2_dict.set_item("rope_length", self.p2.rope_length).unwrap();

        d.set_item("p1", p1_dict).unwrap();
        d.set_item("p2", p2_dict).unwrap();
        d.into()
    }
}

#[pymodule]
fn wdb_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEnv>()?;
    Ok(())
}