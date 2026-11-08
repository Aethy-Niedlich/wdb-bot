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

    fn reset(&mut self, seed: Option<u64>) -> PyResult<PyObject> {
        self.state = GameState::reset(seed.unwrap_or(0));
        Python::with_gil(|py| self.state.to_py(py))
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

    /// Bit-exact stepping from a JSON state + input — used by the parity test.
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
        let d = PyDict::new(py);
        // fill the same keys your ObservationEncoder already expects
        d.set_item("t", self.t).unwrap();
        // ...
        d.into()
    }
}

#[pymodule]
fn wdb_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEnv>()?;
    Ok(())
}
