// Helper to convert serde_json::Value to Py<PyAny>
fn json_to_pyobject(py: Python, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => {
            let py_bool = PyBool::new(py, *b);
            Ok(py_bool.as_any().clone().unbind())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let py_int = PyInt::new(py, i);
                Ok(py_int.as_any().clone().unbind())
            } else if let Some(f) = n.as_f64() {
                let py_float = PyFloat::new(py, f);
                Ok(py_float.as_any().clone().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => {
            let py_str = PyString::new(py, s);
            Ok(py_str.as_any().clone().unbind())
        }
        serde_json::Value::Array(arr) => {
            let py_list: Vec<Py<PyAny>> = arr
                .iter()
                .map(|v| json_to_pyobject(py, v))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, py_list)?.as_any().clone().unbind())
        }
        serde_json::Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (k, v) in obj {
                py_dict.set_item(k, json_to_pyobject(py, v)?)?;
            }
            Ok(py_dict.as_any().clone().unbind())
        }
    }
}
