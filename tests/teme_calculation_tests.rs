use chrono::{DateTime, Utc};
use sgp4::{parse_2les, Constants};

/// Test TLE for NORAD ID 28485 (2004-047A)
/// Epoch: 2025-10-14T13:37:10.647840 UTC
const TEST_TLE1: &str = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995";
const TEST_TLE2: &str = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530";

// Orbital distance bounds for validation (km)
const MIN_ORBITAL_DISTANCE_KM: f64 = 6500.0; // Low Earth Orbit lower bound
const MAX_ORBITAL_DISTANCE_KM: f64 = 45000.0; // Beyond Geostationary Orbit

// Fixture-like helper: Returns parsed elements and constants
fn setup_elements_and_constants() -> (sgp4::Elements, Constants) {
    let tle_string = format!("{}\n{}\n", TEST_TLE1, TEST_TLE2);
    let elements_vec = parse_2les(&tle_string).expect("Failed to parse TLE");
    let elements = elements_vec.into_iter().next().unwrap();
    let constants = Constants::from_elements(&elements).expect("Failed to create constants");
    (elements, constants)
}

// Helper for propagation at a specific time
fn propagate_at_time(
    elements: &sgp4::Elements,
    constants: &Constants,
    time_str: &str,
) -> sgp4::Prediction {
    let test_time = DateTime::parse_from_rfc3339(time_str)
        .expect("Failed to parse test time")
        .with_timezone(&Utc);
    let naive_dt = test_time.naive_utc();
    let minutes_since_epoch = elements
        .datetime_to_minutes_since_epoch(&naive_dt)
        .expect("Failed to calculate minutes");
    constants
        .propagate(minutes_since_epoch)
        .expect("Failed to propagate")
}

#[test]
fn test_parse_tle_elements_len() {
    let tle_string = format!("{}\n{}\n", TEST_TLE1, TEST_TLE2);
    let elements_vec = parse_2les(&tle_string).expect("Failed to parse TLE");
    assert_eq!(
        elements_vec.len(),
        1,
        "Should parse exactly one element set"
    );
}

#[test]
fn test_parse_tle_norad_id() {
    let (elements, _) = setup_elements_and_constants();
    assert_eq!(elements.norad_id, 28485);
}

#[test]
fn test_parse_tle_international_designator() {
    let (elements, _) = setup_elements_and_constants();
    assert_eq!(
        elements.international_designator,
        Some("2004-047A".to_string())
    );
}

#[test]
fn test_datetime_to_minutes_negative() {
    let (elements, _) = setup_elements_and_constants();
    let test_time = DateTime::parse_from_rfc3339("2025-10-14T12:00:00Z")
        .expect("Failed to parse test time")
        .with_timezone(&Utc);
    let naive_dt = test_time.naive_utc();
    let minutes = elements
        .datetime_to_minutes_since_epoch(&naive_dt)
        .expect("Failed to calculate minutes since epoch");
    assert!(minutes.0 < 0.0, "Minutes should be negative (before epoch)");
}

#[test]
fn test_datetime_to_minutes_value() {
    let (elements, _) = setup_elements_and_constants();
    let test_time = DateTime::parse_from_rfc3339("2025-10-14T12:00:00Z")
        .expect("Failed to parse test time")
        .with_timezone(&Utc);
    let naive_dt = test_time.naive_utc();
    let minutes = elements
        .datetime_to_minutes_since_epoch(&naive_dt)
        .expect("Failed to calculate minutes since epoch");
    assert!(
        (minutes.0 + 97.18).abs() < 0.1,
        "Expected approximately -97.18 minutes, got {}",
        minutes.0
    );
}

#[test]
fn test_propagate_position_distance_min() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance > MIN_ORBITAL_DISTANCE_KM,
        "Position distance {} km is below minimum orbital distance",
        distance
    );
}

#[test]
fn test_propagate_position_distance_max() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance < MAX_ORBITAL_DISTANCE_KM,
        "Position distance {} km is above maximum orbital distance",
        distance
    );
}

#[test]
fn test_propagate_position_x_nonzero() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos = prediction.position;
    assert!(pos[0].abs() > 0.1, "X position should be non-zero");
}

#[test]
fn test_propagate_position_y_nonzero() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos = prediction.position;
    assert!(pos[1].abs() > 0.1, "Y position should be non-zero");
}

#[test]
fn test_propagate_position_z_nonzero() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos = prediction.position;
    assert!(pos[2].abs() > 0.1, "Z position should be non-zero");
}

#[test]
fn test_propagate_velocity_speed_min() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let vel = prediction.velocity;
    let speed = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
    assert!(
        speed > 1.0,
        "Velocity magnitude {} km/s is below minimum",
        speed
    );
}

#[test]
fn test_propagate_velocity_speed_max() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let vel = prediction.velocity;
    let speed = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
    assert!(
        speed < 12.0,
        "Velocity magnitude {} km/s is above maximum",
        speed
    );
}

#[test]
fn test_teme_x_match() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let expected_x = -4148.4;
    let tolerance_km = 1.0;
    assert!(
        (prediction.position[0] - expected_x).abs() < tolerance_km,
        "X position {} differs from expected {} by more than {} km",
        prediction.position[0],
        expected_x,
        tolerance_km
    );
}

#[test]
fn test_teme_y_match() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let expected_y = -5360.8;
    let tolerance_km = 1.0;
    assert!(
        (prediction.position[1] - expected_y).abs() < tolerance_km,
        "Y position {} differs from expected {} by more than {} km",
        prediction.position[1],
        expected_y,
        tolerance_km
    );
}

#[test]
fn test_teme_z_match() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let expected_z = -667.7;
    let tolerance_km = 1.0;
    assert!(
        (prediction.position[2] - expected_z).abs() < tolerance_km,
        "Z position {} differs from expected {} by more than {} km",
        prediction.position[2],
        expected_z,
        tolerance_km
    );
}

#[test]
fn test_propagation_at_12_00_distance_min() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance > MIN_ORBITAL_DISTANCE_KM,
        "Distance {} km at 2025-10-14T12:00:00Z is below minimum",
        distance
    );
}

#[test]
fn test_propagation_at_12_00_distance_max() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance < MAX_ORBITAL_DISTANCE_KM,
        "Distance {} km at 2025-10-14T12:00:00Z is above maximum",
        distance
    );
}

#[test]
fn test_propagation_at_12_10_distance_min() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:10:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance > MIN_ORBITAL_DISTANCE_KM,
        "Distance {} km at 2025-10-14T12:10:00Z is below minimum",
        distance
    );
}

#[test]
fn test_propagation_at_12_10_distance_max() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T12:10:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance < MAX_ORBITAL_DISTANCE_KM,
        "Distance {} km at 2025-10-14T12:10:00Z is above maximum",
        distance
    );
}

#[test]
fn test_propagation_at_13_00_distance_min() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T13:00:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance > MIN_ORBITAL_DISTANCE_KM,
        "Distance {} km at 2025-10-14T13:00:00Z is below minimum",
        distance
    );
}

#[test]
fn test_propagation_at_13_00_distance_max() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction = propagate_at_time(&elements, &constants, "2025-10-14T13:00:00Z");
    let pos = prediction.position;
    let distance = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        distance < MAX_ORBITAL_DISTANCE_KM,
        "Distance {} km at 2025-10-14T13:00:00Z is above maximum",
        distance
    );
}

#[test]
fn test_propagation_distances_differ_12_00_vs_12_10() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction1 = propagate_at_time(&elements, &constants, "2025-10-14T12:00:00Z");
    let pos1 = prediction1.position;
    let distance1 = (pos1[0] * pos1[0] + pos1[1] * pos1[1] + pos1[2] * pos1[2]).sqrt();

    let prediction2 = propagate_at_time(&elements, &constants, "2025-10-14T12:10:00Z");
    let pos2 = prediction2.position;
    let distance2 = (pos2[0] * pos2[0] + pos2[1] * pos2[1] + pos2[2] * pos2[2]).sqrt();

    assert_ne!(
        distance1, distance2,
        "Distance should differ between 12:00 and 12:10"
    );
}

#[test]
fn test_propagation_distances_differ_12_10_vs_13_00() {
    let (elements, constants) = setup_elements_and_constants();
    let prediction1 = propagate_at_time(&elements, &constants, "2025-10-14T12:10:00Z");
    let pos1 = prediction1.position;
    let distance1 = (pos1[0] * pos1[0] + pos1[1] * pos1[1] + pos1[2] * pos1[2]).sqrt();

    let prediction2 = propagate_at_time(&elements, &constants, "2025-10-14T13:00:00Z");
    let pos2 = prediction2.position;
    let distance2 = (pos2[0] * pos2[0] + pos2[1] * pos2[1] + pos2[2] * pos2[2]).sqrt();

    assert_ne!(
        distance1, distance2,
        "Distance should differ between 12:10 and 13:00"
    );
}
