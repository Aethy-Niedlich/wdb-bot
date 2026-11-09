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

    fn step(&mut self, action: Vec<i8>) -> PyResult<(PyObject, f64, bool)> {
        let inputs = Inputs {
            move_dir: action[0] - 1,
            jump_dir: action[1] - 1,
            heavy: action[2] == 1,
        };
        step(&mut self.state, &inputs);
        let reward = self.state.reward();
        let done = self.state.is_done();
        Python::with_gil(|py| Ok((self.state.to_py(py), reward, done)))
    }

    fn step_json(&mut self, state_json: &str, action: Vec<i8>) -> PyResult<String> {
        let mut s: GameState = serde_json::from_str(state_json).unwrap();
        let inputs = Inputs {
            move_dir: action[0] - 1,
            jump_dir: action[1] - 1,
            heavy: action[2] == 1,
        };
        step(&mut s, &inputs);
        Ok(serde_json::to_string(&s).unwrap())
    }
}

impl GameState {
    fn to_py(&self, py: Python) -> PyObject {
        let d = PyDict::new_bound(py); 
        d.set_item("t", self.t).unwrap();
        d.set_item("x", self.x).unwrap();
        d.set_item("y", self.y).unwrap();
        d.set_item("vx", self.vx).unwrap();
        d.set_item("vy", self.vy).unwrap();
        d.set_item("force_x", self.force_x).unwrap();
        d.set_item("force_y", self.force_y).unwrap();
        d.set_item("mass", self.mass).unwrap();
        d.set_item("inv_mass", self.inv_mass).unwrap();
        d.set_item("linear_damping", self.linear_damping).unwrap();
        d.set_item("angle", self.angle).unwrap();
        d.set_item("angular_velocity", self.angular_velocity).unwrap();
        d.set_item("torque", self.torque).unwrap();
        d.set_item("inertia", self.inertia).unwrap();
        d.set_item("inv_inertia", self.inv_inertia).unwrap();
        d.set_item("angular_damping", self.angular_damping).unwrap();
        d.into()
    }
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEnv>()?;
    Ok(())
}