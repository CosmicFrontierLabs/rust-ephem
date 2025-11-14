use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime, types::PyModule};

/// Configuration for converting position/velocity data to astropy SkyCoord
pub struct SkyCoordConfig<'a> {
    /// Position and velocity data (N x 6: [x, y, z, vx, vy, vz])
    pub data: &'a Array2<f64>,
    /// Time objects as Python datetime
    pub time_objects: Vec<Py<PyDateTime>>,
    /// Coordinate frame name (e.g., "GCRS", "ITRS")
    pub frame_name: &'a str,
    /// Whether to negate the position and velocity vectors
    pub negate_vectors: bool,
    /// Optional observer position and velocity data (N x 6: [x, y, z, vx, vy, vz])
    /// Used to set both obsgeoloc and obsgeovel frame parameters
    pub observer_data: Option<&'a Array2<f64>>,
}

/// Cached Python modules for astropy to avoid repeated imports
pub struct AstropyModules<'py> {
    pub coords: Bound<'py, PyModule>,
    pub time: Bound<'py, PyModule>,
    pub units: Bound<'py, PyModule>,
}

impl<'py> AstropyModules<'py> {
    /// Import astropy modules once for reuse
    pub fn import(py: Python<'py>) -> PyResult<Self> {
        Ok(AstropyModules {
            coords: py.import("astropy.coordinates")?,
            time: py.import("astropy.time")?,
            units: py.import("astropy.units")?,
        })
    }
}

/// Convert position and velocity data to an astropy SkyCoord object
///
/// This is a generic function that can create SkyCoord objects for different
/// coordinate frames (GCRS, ITRS) and handle optional observer location/velocity
/// parameters (obsgeoloc/obsgeovel).
///
/// # Arguments
/// * `py` - Python interpreter
/// * `modules` - Pre-imported astropy modules (pass None to import on-demand)
/// * `config` - Configuration specifying the data and frame parameters
///
/// # Returns
/// A Python SkyCoord object with the specified coordinate frame
pub fn to_skycoord(
    py: Python,
    modules: Option<&AstropyModules>,
    config: SkyCoordConfig,
) -> PyResult<Py<PyAny>> {
    // Import astropy modules if not provided (for backward compatibility)
    let temp_modules;
    let astropy_modules = if let Some(m) = modules {
        m
    } else {
        temp_modules = AstropyModules::import(py)?;
        &temp_modules
    };

    let astropy_coords = &astropy_modules.coords;
    let astropy_time = &astropy_modules.time;
    let astropy_units = &astropy_modules.units;

    // Get position and velocity arrays
    let positions = config.data.slice(ndarray::s![.., 0..3]).to_owned();
    let velocities = config.data.slice(ndarray::s![.., 3..6]).to_owned();

    // Apply negation if requested (for Earth-relative coordinates)
    let (final_positions, final_velocities) = if config.negate_vectors {
        (-positions, -velocities)
    } else {
        (positions, velocities)
    };

    // Convert to numpy arrays
    let pos_array = final_positions.into_pyarray(py);
    let vel_array = final_velocities.into_pyarray(py);

    // Get units (km and km/s)
    let km_unit = astropy_units.getattr("km")?;
    let s_unit = astropy_units.getattr("s")?;
    let km_per_s_unit = km_unit.call_method1("__truediv__", (s_unit,))?;

    // Multiply arrays by units using __rmul__ on the unit side
    let pos_with_unit = km_unit.call_method1("__rmul__", (pos_array,))?;
    let vel_with_unit = km_per_s_unit.call_method1("__rmul__", (vel_array,))?;

    // Create Time object from Python datetime objects
    let time_dict = pyo3::types::PyDict::new(py);
    time_dict.set_item("scale", "utc")?;
    let time_obj = astropy_time
        .getattr("Time")?
        .call((config.time_objects,), Some(&time_dict))?;

    // Create frame with obstime and optional obsgeoloc/obsgeovel
    let frame_dict = pyo3::types::PyDict::new(py);
    frame_dict.set_item("obstime", time_obj)?;

    // Add obsgeoloc and obsgeovel if observer data is provided
    if let Some(observer_data) = config.observer_data {
        // Get observer position and velocity arrays
        let observer_positions = observer_data.slice(ndarray::s![.., 0..3]).to_owned();
        let observer_velocities = observer_data.slice(ndarray::s![.., 3..6]).to_owned();

        // Convert to numpy arrays
        let obs_pos_array = observer_positions.into_pyarray(py);
        let obs_vel_array = observer_velocities.into_pyarray(py);

        // Apply units
        let obs_pos_with_unit = km_unit.call_method1("__rmul__", (obs_pos_array,))?;
        let obs_vel_with_unit = km_per_s_unit.call_method1("__rmul__", (obs_vel_array,))?;

        // Extract observer position components
        let obs_x = obs_pos_with_unit
            .getattr("T")?
            .getattr("__getitem__")?
            .call1((0,))?;
        let obs_y = obs_pos_with_unit
            .getattr("T")?
            .getattr("__getitem__")?
            .call1((1,))?;
        let obs_z = obs_pos_with_unit
            .getattr("T")?
            .getattr("__getitem__")?
            .call1((2,))?;

        let obs_v_x = obs_vel_with_unit
            .getattr("T")?
            .getattr("__getitem__")?
            .call1((0,))?;
        let obs_v_y = obs_vel_with_unit
            .getattr("T")?
            .getattr("__getitem__")?
            .call1((1,))?;
        let obs_v_z = obs_vel_with_unit
            .getattr("T")?
            .getattr("__getitem__")?
            .call1((2,))?;

        // Create CartesianRepresentation for obsgeoloc
        let obsgeoloc_dict = pyo3::types::PyDict::new(py);
        obsgeoloc_dict.set_item("x", obs_x)?;
        obsgeoloc_dict.set_item("y", obs_y)?;
        obsgeoloc_dict.set_item("z", obs_z)?;
        let obsgeoloc_rep = astropy_coords
            .getattr("CartesianRepresentation")?
            .call((), Some(&obsgeoloc_dict))?;

        // Create CartesianRepresentation for obsgeovel
        let obsgeovel_dict = pyo3::types::PyDict::new(py);
        obsgeovel_dict.set_item("x", obs_v_x)?;
        obsgeovel_dict.set_item("y", obs_v_y)?;
        obsgeovel_dict.set_item("z", obs_v_z)?;
        let obsgeovel_rep = astropy_coords
            .getattr("CartesianRepresentation")?
            .call((), Some(&obsgeovel_dict))?;

        frame_dict.set_item("obsgeoloc", obsgeoloc_rep)?;
        frame_dict.set_item("obsgeovel", obsgeovel_rep)?;
    }

    // Get the frame class by name and instantiate it
    let frame = astropy_coords
        .getattr(config.frame_name)?
        .call((), Some(&frame_dict))?;

    // Extract x, y, z components
    let x = pos_with_unit
        .getattr("T")?
        .getattr("__getitem__")?
        .call1((0,))?;
    let y = pos_with_unit
        .getattr("T")?
        .getattr("__getitem__")?
        .call1((1,))?;
    let z = pos_with_unit
        .getattr("T")?
        .getattr("__getitem__")?
        .call1((2,))?;

    let v_x = vel_with_unit
        .getattr("T")?
        .getattr("__getitem__")?
        .call1((0,))?;
    let v_y = vel_with_unit
        .getattr("T")?
        .getattr("__getitem__")?
        .call1((1,))?;
    let v_z = vel_with_unit
        .getattr("T")?
        .getattr("__getitem__")?
        .call1((2,))?;

    // Create CartesianDifferential for velocity
    let diff_dict = pyo3::types::PyDict::new(py);
    diff_dict.set_item("d_x", v_x)?;
    diff_dict.set_item("d_y", v_y)?;
    diff_dict.set_item("d_z", v_z)?;
    let cart_diff = astropy_coords
        .getattr("CartesianDifferential")?
        .call((), Some(&diff_dict))?;

    // Create CartesianRepresentation with differential
    let rep_dict = pyo3::types::PyDict::new(py);
    rep_dict.set_item("x", x)?;
    rep_dict.set_item("y", y)?;
    rep_dict.set_item("z", z)?;
    rep_dict.set_item("differentials", cart_diff)?;
    let cart_rep = astropy_coords
        .getattr("CartesianRepresentation")?
        .call((), Some(&rep_dict))?;

    // Create SkyCoord with CartesianRepresentation and frame
    let coord_dict = pyo3::types::PyDict::new(py);
    coord_dict.set_item("frame", frame)?;

    let skycoord = astropy_coords
        .getattr("SkyCoord")?
        .call((cart_rep,), Some(&coord_dict))?;

    Ok(skycoord.into())
}
