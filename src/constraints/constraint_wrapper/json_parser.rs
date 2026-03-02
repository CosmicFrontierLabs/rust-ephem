// Helper function to parse constraint JSON into evaluator
fn parse_constraint_json(value: &serde_json::Value) -> PyResult<Box<dyn ConstraintEvaluator>> {
    let constraint_type = value.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing or invalid 'type' field in JSON")
    })?;

    match constraint_type {
        "sun" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let config = SunProximityConfig {
                min_angle,
                max_angle,
            };
            Ok(config.to_evaluator())
        }
        "moon" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let config = MoonProximityConfig {
                min_angle,
                max_angle,
            };
            Ok(config.to_evaluator())
        }
        "eclipse" => {
            let umbra_only = value
                .get("umbra_only")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let config = EclipseConfig { umbra_only };
            Ok(config.to_evaluator())
        }
        "earth_limb" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let include_refraction = value
                .get("include_refraction")
                .and_then(|v| v.as_bool())
                .unwrap_or(false); // Default to false if not specified
            let horizon_dip = value
                .get("horizon_dip")
                .and_then(|v| v.as_bool())
                .unwrap_or(false); // Default to false if not specified
            let config = EarthLimbConfig {
                min_angle,
                max_angle,
                include_refraction,
                horizon_dip,
            };
            Ok(config.to_evaluator())
        }
        "body" => {
            let body = value
                .get("body")
                .and_then(|v| v.as_str())
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'body' field"))?
                .to_string();
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let config = BodyProximityConfig {
                body,
                min_angle,
                max_angle,
            };
            Ok(config.to_evaluator())
        }
        "daytime" => {
            let twilight = value
                .get("twilight")
                .and_then(|v| v.as_str())
                .unwrap_or("civil");
            let twilight_type = match twilight {
                "civil" => TwilightType::Civil,
                "nautical" => TwilightType::Nautical,
                "astronomical" => TwilightType::Astronomical,
                "none" => TwilightType::None,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown twilight type: {twilight}"
                    )))
                }
            };
            let config = DaytimeConfig {
                twilight: twilight_type,
            };
            Ok(config.to_evaluator())
        }
        "airmass" => {
            let max_airmass = value
                .get("max_airmass")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'max_airmass' field")
                })?;
            let min_airmass = value.get("min_airmass").and_then(|v| v.as_f64());
            let config = AirmassConfig {
                min_airmass,
                max_airmass,
            };
            Ok(config.to_evaluator())
        }
        "moon_phase" => {
            let max_illumination = value
                .get("max_illumination")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'max_illumination' field")
                })?;
            let min_illumination = value.get("min_illumination").and_then(|v| v.as_f64());
            let min_distance = value.get("min_distance").and_then(|v| v.as_f64());
            let max_distance = value.get("max_distance").and_then(|v| v.as_f64());
            let enforce_when_below_horizon = value
                .get("enforce_when_below_horizon")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let moon_visibility = value
                .get("moon_visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("full")
                .to_string();
            let config = MoonPhaseConfig {
                min_illumination,
                max_illumination,
                min_distance,
                max_distance,
                enforce_when_below_horizon,
                moon_visibility,
            };
            Ok(config.to_evaluator())
        }
        "saa" => {
            let polygon = value
                .get("polygon")
                .and_then(|v| v.as_array())
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'polygon' field"))?
                .iter()
                .map(|point| {
                    let arr = point.as_array().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Polygon points must be arrays")
                    })?;
                    if arr.len() != 2 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Polygon points must be [lon, lat] pairs",
                        ));
                    }
                    let lon = arr[0].as_f64().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Longitude must be a number")
                    })?;
                    let lat = arr[1].as_f64().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Latitude must be a number")
                    })?;
                    Ok((lon, lat))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let config = SAAConfig { polygon };
            Ok(config.to_evaluator())
        }
        "alt_az" => {
            let min_altitude = value.get("min_altitude").and_then(|v| v.as_f64());
            let max_altitude = value.get("max_altitude").and_then(|v| v.as_f64());
            let min_azimuth = value.get("min_azimuth").and_then(|v| v.as_f64());
            let max_azimuth = value.get("max_azimuth").and_then(|v| v.as_f64());
            let polygon = value
                .get("polygon")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|point| {
                            let p = point.as_array().ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err(
                                    "Polygon points must be arrays",
                                )
                            })?;
                            if p.len() != 2 {
                                return Err(pyo3::exceptions::PyValueError::new_err(
                                    "Polygon points must be [alt, az] pairs",
                                ));
                            }
                            let alt = p[0].as_f64().ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err("Altitude must be a number")
                            })?;
                            let az = p[1].as_f64().ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err("Azimuth must be a number")
                            })?;
                            Ok((alt, az))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            let config = AltAzConfig {
                min_altitude,
                max_altitude,
                min_azimuth,
                max_azimuth,
                polygon,
            };
            Ok(config.to_evaluator())
        }
        "and" => {
            let constraints = value
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'constraints' array for AND")
                })?;
            let evaluators: Result<Vec<_>, _> =
                constraints.iter().map(parse_constraint_json).collect();
            Ok(Box::new(AndEvaluator {
                constraints: evaluators?,
            }))
        }
        "or" => {
            let constraints = value
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'constraints' array for OR")
                })?;
            let evaluators: Result<Vec<_>, _> =
                constraints.iter().map(parse_constraint_json).collect();
            Ok(Box::new(OrEvaluator {
                constraints: evaluators?,
            }))
        }
        "xor" => {
            let constraints = value
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'constraints' array for XOR")
                })?;
            if constraints.len() < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "XOR requires at least two sub-constraints",
                ));
            }
            let evaluators: Result<Vec<_>, _> =
                constraints.iter().map(parse_constraint_json).collect();
            Ok(Box::new(XorEvaluator {
                constraints: evaluators?,
            }))
        }
        "at_least" => {
            let constraints = value
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Missing 'constraints' array for at_least",
                    )
                })?;
            if constraints.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "at_least requires at least one sub-constraint",
                ));
            }

            let min_violated = value
                .get("min_violated")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Missing or invalid 'min_violated' field for at_least",
                    )
                })? as usize;

            if min_violated == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_violated must be at least 1",
                ));
            }
            if min_violated > constraints.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "min_violated ({min_violated}) cannot exceed number of constraints ({})",
                    constraints.len()
                )));
            }

            let evaluators: Result<Vec<_>, _> =
                constraints.iter().map(parse_constraint_json).collect();
            Ok(Box::new(AtLeastEvaluator {
                constraints: evaluators?,
                min_violated,
            }))
        }
        "not" => {
            let constraint = value.get("constraint").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'constraint' field for NOT")
            })?;
            let evaluator = parse_constraint_json(constraint)?;
            Ok(Box::new(NotEvaluator {
                constraint: evaluator,
            }))
        }
        "boresight_offset" => {
            let constraint = value.get("constraint").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "Missing 'constraint' field for boresight_offset",
                )
            })?;
            let evaluator = parse_constraint_json(constraint)?;
            let roll_deg = value
                .get("roll_deg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let pitch_deg = value
                .get("pitch_deg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let yaw_deg = value.get("yaw_deg").and_then(|v| v.as_f64()).unwrap_or(0.0);

            if !roll_deg.is_finite() || !pitch_deg.is_finite() || !yaw_deg.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "roll_deg, pitch_deg, and yaw_deg must be finite numbers",
                ));
            }

            let rotation_matrix =
                crate::utils::vector_math::euler_zyx_rotation_matrix(roll_deg, pitch_deg, yaw_deg);

            Ok(Box::new(BoresightOffsetEvaluator {
                constraint: evaluator,
                roll_deg,
                pitch_deg,
                yaw_deg,
                rotation_matrix,
            }))
        }
        "orbit_pole" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let earth_limb_pole = value
                .get("earth_limb_pole")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let config = OrbitPoleConfig {
                min_angle,
                max_angle,
                earth_limb_pole,
            };
            Ok(config.to_evaluator())
        }
        "orbit_ram" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let config = OrbitRamConfig {
                min_angle,
                max_angle,
            };
            Ok(config.to_evaluator())
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown constraint type: {constraint_type}"
        ))),
    }
}
