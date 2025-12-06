// Module declarations
mod constraints;
mod ephemeris;
mod utils;

// Re-export public API from ephemeris
pub use ephemeris::position_velocity::PositionVelocityData;
pub use ephemeris::{GroundEphemeris, OEMEphemeris, SPICEEphemeris, TLEEphemeris};

// Re-export constraint types
pub use constraints::{ConstraintResult, ConstraintViolation, PyConstraint, VisibilityWindow};

// Make certain utils modules public for external access
pub use utils::{eop_provider, naif_ids, ut1_provider};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::utils::config::{DE440S_URL, DE440_URL};
use crate::utils::config::{DEFAULT_DE440S_PATH, DEFAULT_DE440_PATH};

#[pyfunction]
fn init_planetary_ephemeris(py_path: String) -> PyResult<()> {
    let p = std::path::Path::new(&py_path);
    ephemeris::spice_manager::init_planetary_ephemeris(p).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to init planetary SPK '{py_path}': {e:?}"
        ))
    })?;
    Ok(())
}

#[pyfunction]
fn download_planetary_ephemeris(url: String, dest: String) -> PyResult<()> {
    let p = std::path::Path::new(&dest);
    ephemeris::spice_manager::download_planetary_ephemeris(&url, p).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to download {url} -> {dest}: {e:?}"
        ))
    })?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (py_path=None, download_if_missing=true, spk_url=None, prefer_full=false))]
fn ensure_planetary_ephemeris(
    py_path: Option<String>,
    download_if_missing: bool,
    spk_url: Option<String>,
    prefer_full: bool,
) -> PyResult<()> {
    use std::path::Path;

    // If already initialized and prefer_full was requested with no explicit path,
    // upgrade to the full kernel when it exists.
    if ephemeris::spice_manager::is_planetary_ephemeris_initialized()
        && py_path.is_none()
        && prefer_full
    {
        let full = DEFAULT_DE440_PATH.as_path();
        if full.exists() {
            // Emit a Python warning about re-initialization
            use pyo3::types::PyModule;
            use pyo3::types::PyString;
            use pyo3::Python;
            Python::attach(|py| {
                let warning_msg = format!(
                    "Upgrading planetary ephemeris to full kernel '{}'. This will re-initialize the ephemeris and may have performance or behavioral implications.",
                    full.display()
                );
                let warnings =
                    PyModule::import(py, "warnings").expect("Failed to import warnings module");
                let _ = warnings.call_method1("warn", (PyString::new(py, &warning_msg),));
            });
            ephemeris::spice_manager::init_planetary_ephemeris(full).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to init full planetary SPK '{}': {:?}",
                    full.display(),
                    e
                ))
            })?;
            // Return early after successful upgrade to full kernel.
            return Ok(());
        }
    }

    // Resolve target path:
    // 1) If explicit path provided, use it.
    // 2) Else prefer full DE440 if it already exists.
    // 3) Else prefer slim DE440S if it exists.
    // 4) Else choose based on prefer_full flag (controls download target and default URL).
    let (path_str, default_url) = if let Some(p) = py_path {
        (p, None::<String>)
    } else {
        let full = DEFAULT_DE440_PATH.as_path();
        if full.exists() {
            (full.to_string_lossy().to_string(), None::<String>)
        } else {
            let slim = DEFAULT_DE440S_PATH.as_path();
            if slim.exists() {
                (slim.to_string_lossy().to_string(), None::<String>)
            } else if prefer_full {
                (
                    DEFAULT_DE440_PATH.to_string_lossy().to_string(),
                    Some(DE440_URL.to_string()),
                )
            } else {
                (
                    DEFAULT_DE440S_PATH.to_string_lossy().to_string(),
                    Some(DE440S_URL.to_string()),
                )
            }
        }
    };
    let path = Path::new(&path_str);

    // Check if file exists before anything else
    if !path.exists() {
        if download_if_missing {
            let url = if let Some(provided) = spk_url {
                provided
            } else if let Some(default) = default_url.clone() {
                default
            } else {
                // Fallback: prefer full if requested, otherwise slim
                if prefer_full {
                    DE440_URL.to_string()
                } else {
                    DE440S_URL.to_string()
                }
            };
            ephemeris::spice_manager::download_planetary_ephemeris(&url, path).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to download planetary SPK from {url}: {e:?}"
                ))
            })?;
        } else {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Planetary SPK file not found: {path_str}"
            )));
        }
    }

    // Check if already initialized (only after confirming file exists)
    if ephemeris::spice_manager::is_planetary_ephemeris_initialized() {
        return Ok(());
    }

    // Initialize the almanac
    ephemeris::spice_manager::ensure_planetary_ephemeris(path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to load planetary SPK '{path_str}': {e:?}"
        ))
    })?;
    Ok(())
}

#[pyfunction]
fn is_planetary_ephemeris_initialized() -> bool {
    ephemeris::spice_manager::is_planetary_ephemeris_initialized()
}

/// Helper function to convert PyDateTime to chrono::DateTime<Utc>
fn pydatetime_to_chrono(
    py_datetime: &Bound<'_, pyo3::types::PyDateTime>,
) -> PyResult<chrono::DateTime<chrono::Utc>> {
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

    let year = py_datetime.getattr("year")?.extract::<i32>()?;
    let month = py_datetime.getattr("month")?.extract::<u32>()?;
    let day = py_datetime.getattr("day")?.extract::<u32>()?;
    let hour = py_datetime.getattr("hour")?.extract::<u32>()?;
    let minute = py_datetime.getattr("minute")?.extract::<u32>()?;
    let second = py_datetime.getattr("second")?.extract::<u32>()?;
    let micro = py_datetime.getattr("microsecond")?.extract::<u32>()?;

    let naive_date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid date"))?;
    let naive_time = NaiveTime::from_hms_micro_opt(hour, minute, second, micro)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid time"))?;
    let naive_dt = NaiveDateTime::new(naive_date, naive_time);
    Ok(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
        naive_dt,
        chrono::Utc,
    ))
}

#[pyfunction]
fn get_tai_utc_offset(py_datetime: &Bound<'_, pyo3::types::PyDateTime>) -> PyResult<Option<f64>> {
    let dt = pydatetime_to_chrono(py_datetime)?;
    Ok(utils::leap_seconds::get_tai_utc_offset(&dt))
}

#[pyfunction]
fn get_ut1_utc_offset(py_datetime: &Bound<'_, pyo3::types::PyDateTime>) -> PyResult<f64> {
    let dt = pydatetime_to_chrono(py_datetime)?;
    Ok(utils::ut1_provider::get_ut1_utc_offset(&dt))
}

#[pyfunction]
fn is_ut1_available() -> bool {
    utils::ut1_provider::is_ut1_available()
}

#[pyfunction]
fn init_ut1_provider() -> bool {
    utils::ut1_provider::init_ut1_provider()
}

#[pyfunction]
fn get_polar_motion(py_datetime: &Bound<'_, pyo3::types::PyDateTime>) -> PyResult<(f64, f64)> {
    let dt = pydatetime_to_chrono(py_datetime)?;
    Ok(utils::eop_provider::get_polar_motion(&dt))
}

#[pyfunction]
fn is_eop_available() -> bool {
    utils::eop_provider::is_eop_available()
}

#[pyfunction]
fn init_eop_provider() -> bool {
    utils::eop_provider::init_eop_provider()
}

/// Returns the cache directory path used by rust_ephem for storing data files
#[pyfunction]
fn get_cache_dir() -> String {
    utils::config::CACHE_DIR.to_string_lossy().to_string()
}

/// Fetch a TLE from various sources (file, URL, Celestrak, Space-Track.org)
///
/// Returns a dict with keys: line1, line2, name (optional), epoch, source
#[pyfunction]
#[pyo3(signature = (*, tle=None, norad_id=None, norad_name=None, epoch=None, spacetrack_username=None, spacetrack_password=None, epoch_tolerance_days=None))]
#[allow(clippy::too_many_arguments)]
fn fetch_tle(
    py: Python,
    tle: Option<String>,
    norad_id: Option<u32>,
    norad_name: Option<String>,
    epoch: Option<&Bound<'_, pyo3::types::PyDateTime>>,
    spacetrack_username: Option<String>,
    spacetrack_password: Option<String>,
    epoch_tolerance_days: Option<f64>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use crate::utils::tle_utils;

    // Convert epoch if provided
    let epoch_chrono =
        epoch.and_then(|te| crate::utils::time_utils::python_datetime_to_utc(te).ok());

    // Build credentials if provided, otherwise try environment
    let credentials = match (&spacetrack_username, &spacetrack_password) {
        (Some(u), Some(p)) => {
            Some(tle_utils::SpaceTrackCredentials::new(u.clone(), p.clone()))
        }
        (None, None) => tle_utils::SpaceTrackCredentials::from_env().ok(),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Both spacetrack_username and spacetrack_password must be provided together, or omit both to use environment variables"
            ))
        }
    };

    // Use the unified fetch function
    let fetched = tle_utils::fetch_tle_unified(
        tle.as_deref(),
        norad_id,
        norad_name.as_deref(),
        epoch_chrono.as_ref(),
        credentials,
        epoch_tolerance_days,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Build the result dictionary
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("line1", fetched.line1)?;
    dict.set_item("line2", fetched.line2)?;
    dict.set_item("name", fetched.name)?;

    // Convert epoch to Python datetime with UTC timezone
    use chrono::{Datelike, Timelike};
    let py_epoch = pyo3::types::PyDateTime::new(
        py,
        fetched.epoch.year(),
        fetched.epoch.month() as u8,
        fetched.epoch.day() as u8,
        fetched.epoch.hour() as u8,
        fetched.epoch.minute() as u8,
        fetched.epoch.second() as u8,
        fetched.epoch.timestamp_subsec_micros(),
        None,
    )?;

    // Add UTC timezone
    let datetime_mod = py.import("datetime")?;
    let utc_tz = datetime_mod.getattr("timezone")?.getattr("utc")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("tzinfo", utc_tz)?;
    let dt_with_tz = py_epoch.call_method("replace", (), Some(&kwargs))?;

    dict.set_item("epoch", dt_with_tz)?;
    dict.set_item("source", fetched.source)?;

    Ok(dict.into())
}

#[pymodule]
fn _rust_ephem(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TLEEphemeris>()?;
    m.add_class::<SPICEEphemeris>()?;
    m.add_class::<OEMEphemeris>()?;
    m.add_class::<GroundEphemeris>()?;
    m.add_class::<PositionVelocityData>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<ConstraintResult>()?;
    m.add_class::<ConstraintViolation>()?;
    m.add_class::<VisibilityWindow>()?;
    m.add_function(wrap_pyfunction!(init_planetary_ephemeris, m)?)?;
    m.add_function(wrap_pyfunction!(download_planetary_ephemeris, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_planetary_ephemeris, m)?)?;
    m.add_function(wrap_pyfunction!(is_planetary_ephemeris_initialized, m)?)?;
    m.add_function(wrap_pyfunction!(get_tai_utc_offset, m)?)?;
    m.add_function(wrap_pyfunction!(get_ut1_utc_offset, m)?)?;
    m.add_function(wrap_pyfunction!(is_ut1_available, m)?)?;
    m.add_function(wrap_pyfunction!(init_ut1_provider, m)?)?;
    m.add_function(wrap_pyfunction!(get_polar_motion, m)?)?;
    m.add_function(wrap_pyfunction!(is_eop_available, m)?)?;
    m.add_function(wrap_pyfunction!(init_eop_provider, m)?)?;
    m.add_function(wrap_pyfunction!(get_cache_dir, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_tle, m)?)?;
    Ok(())
}
