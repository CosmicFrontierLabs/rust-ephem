use serde::Deserialize;

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_civil() -> String {
    "civil".to_string()
}

fn default_full() -> String {
    "full".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ConstraintSpec {
    #[serde(rename = "sun")]
    Sun {
        min_angle: f64,
        max_angle: Option<f64>,
    },
    #[serde(rename = "moon")]
    Moon {
        min_angle: f64,
        max_angle: Option<f64>,
    },
    #[serde(rename = "eclipse")]
    Eclipse {
        #[serde(default = "default_true")]
        umbra_only: bool,
    },
    #[serde(rename = "earth_limb")]
    EarthLimb {
        min_angle: f64,
        max_angle: Option<f64>,
        #[serde(default = "default_false")]
        include_refraction: bool,
        #[serde(default = "default_false")]
        horizon_dip: bool,
    },
    #[serde(rename = "body")]
    Body {
        body: String,
        min_angle: f64,
        max_angle: Option<f64>,
    },
    #[serde(rename = "daytime")]
    Daytime {
        #[serde(default = "default_civil")]
        twilight: String,
    },
    #[serde(rename = "airmass")]
    Airmass {
        min_airmass: Option<f64>,
        max_airmass: f64,
    },
    #[serde(rename = "moon_phase")]
    MoonPhase {
        min_illumination: Option<f64>,
        max_illumination: f64,
        min_distance: Option<f64>,
        max_distance: Option<f64>,
        #[serde(default = "default_false")]
        enforce_when_below_horizon: bool,
        #[serde(default = "default_full")]
        moon_visibility: String,
    },
    #[serde(rename = "saa")]
    #[allow(clippy::upper_case_acronyms)]
    SAA { polygon: Vec<(f64, f64)> },
    #[serde(rename = "alt_az")]
    AltAz {
        min_altitude: Option<f64>,
        max_altitude: Option<f64>,
        min_azimuth: Option<f64>,
        max_azimuth: Option<f64>,
        polygon: Option<Vec<(f64, f64)>>,
    },
    #[serde(rename = "and")]
    And { constraints: Vec<ConstraintSpec> },
    #[serde(rename = "or")]
    Or { constraints: Vec<ConstraintSpec> },
    #[serde(rename = "xor")]
    Xor { constraints: Vec<ConstraintSpec> },
    #[serde(rename = "at_least")]
    AtLeast {
        min_violated: usize,
        constraints: Vec<ConstraintSpec>,
    },
    #[serde(rename = "not")]
    Not { constraint: Box<ConstraintSpec> },
    #[serde(rename = "boresight_offset")]
    BoresightOffset {
        constraint: Box<ConstraintSpec>,
        #[serde(default)]
        roll_deg: f64,
        #[serde(default)]
        pitch_deg: f64,
        #[serde(default)]
        yaw_deg: f64,
    },
    #[serde(rename = "orbit_pole")]
    OrbitPole {
        min_angle: f64,
        max_angle: Option<f64>,
        #[serde(default = "default_false")]
        earth_limb_pole: bool,
    },
    #[serde(rename = "orbit_ram")]
    OrbitRam {
        min_angle: f64,
        max_angle: Option<f64>,
    },
}

impl ConstraintSpec {
    fn into_evaluator(self) -> PyResult<Box<dyn ConstraintEvaluator>> {
        match self {
            ConstraintSpec::Sun {
                min_angle,
                max_angle,
            } => Ok(SunProximityConfig {
                min_angle,
                max_angle,
            }
            .to_evaluator()),
            ConstraintSpec::Moon {
                min_angle,
                max_angle,
            } => Ok(MoonProximityConfig {
                min_angle,
                max_angle,
            }
            .to_evaluator()),
            ConstraintSpec::Eclipse { umbra_only } => {
                Ok(EclipseConfig { umbra_only }.to_evaluator())
            }
            ConstraintSpec::EarthLimb {
                min_angle,
                max_angle,
                include_refraction,
                horizon_dip,
            } => Ok(EarthLimbConfig {
                min_angle,
                max_angle,
                include_refraction,
                horizon_dip,
            }
            .to_evaluator()),
            ConstraintSpec::Body {
                body,
                min_angle,
                max_angle,
            } => Ok(BodyProximityConfig {
                body,
                min_angle,
                max_angle,
            }
            .to_evaluator()),
            ConstraintSpec::Daytime { twilight } => {
                let twilight_type = match twilight.as_str() {
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
                Ok(DaytimeConfig {
                    twilight: twilight_type,
                }
                .to_evaluator())
            }
            ConstraintSpec::Airmass {
                min_airmass,
                max_airmass,
            } => Ok(AirmassConfig {
                min_airmass,
                max_airmass,
            }
            .to_evaluator()),
            ConstraintSpec::MoonPhase {
                min_illumination,
                max_illumination,
                min_distance,
                max_distance,
                enforce_when_below_horizon,
                moon_visibility,
            } => Ok(MoonPhaseConfig {
                min_illumination,
                max_illumination,
                min_distance,
                max_distance,
                enforce_when_below_horizon,
                moon_visibility,
            }
            .to_evaluator()),
            ConstraintSpec::SAA { polygon } => Ok(SAAConfig { polygon }.to_evaluator()),
            ConstraintSpec::AltAz {
                min_altitude,
                max_altitude,
                min_azimuth,
                max_azimuth,
                polygon,
            } => Ok(AltAzConfig {
                min_altitude,
                max_altitude,
                min_azimuth,
                max_azimuth,
                polygon,
            }
            .to_evaluator()),
            ConstraintSpec::And { constraints } => {
                if constraints.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Missing 'constraints' array for AND",
                    ));
                }
                let evaluators: Result<Vec<_>, _> = constraints
                    .into_iter()
                    .map(ConstraintSpec::into_evaluator)
                    .collect();
                Ok(Box::new(AndEvaluator {
                    constraints: evaluators?,
                }))
            }
            ConstraintSpec::Or { constraints } => {
                if constraints.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Missing 'constraints' array for OR",
                    ));
                }
                let evaluators: Result<Vec<_>, _> = constraints
                    .into_iter()
                    .map(ConstraintSpec::into_evaluator)
                    .collect();
                Ok(Box::new(OrEvaluator {
                    constraints: evaluators?,
                }))
            }
            ConstraintSpec::Xor { constraints } => {
                if constraints.len() < 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "XOR requires at least two sub-constraints",
                    ));
                }
                let evaluators: Result<Vec<_>, _> = constraints
                    .into_iter()
                    .map(ConstraintSpec::into_evaluator)
                    .collect();
                Ok(Box::new(XorEvaluator {
                    constraints: evaluators?,
                }))
            }
            ConstraintSpec::AtLeast {
                min_violated,
                constraints,
            } => {
                if constraints.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "at_least requires at least one sub-constraint",
                    ));
                }
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

                let evaluators: Result<Vec<_>, _> = constraints
                    .into_iter()
                    .map(ConstraintSpec::into_evaluator)
                    .collect();
                Ok(Box::new(AtLeastEvaluator {
                    constraints: evaluators?,
                    min_violated,
                }))
            }
            ConstraintSpec::Not { constraint } => Ok(Box::new(NotEvaluator {
                constraint: constraint.into_evaluator()?,
            })),
            ConstraintSpec::BoresightOffset {
                constraint,
                roll_deg,
                pitch_deg,
                yaw_deg,
            } => {
                if !roll_deg.is_finite() || !pitch_deg.is_finite() || !yaw_deg.is_finite() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "roll_deg, pitch_deg, and yaw_deg must be finite numbers",
                    ));
                }

                let rotation_matrix = crate::utils::vector_math::euler_zyx_rotation_matrix(
                    roll_deg, pitch_deg, yaw_deg,
                );

                Ok(Box::new(BoresightOffsetEvaluator {
                    constraint: constraint.into_evaluator()?,
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                    rotation_matrix,
                }))
            }
            ConstraintSpec::OrbitPole {
                min_angle,
                max_angle,
                earth_limb_pole,
            } => Ok(OrbitPoleConfig {
                min_angle,
                max_angle,
                earth_limb_pole,
            }
            .to_evaluator()),
            ConstraintSpec::OrbitRam {
                min_angle,
                max_angle,
            } => Ok(OrbitRamConfig {
                min_angle,
                max_angle,
            }
            .to_evaluator()),
        }
    }
}

// Helper function to parse constraint JSON into evaluator
fn parse_constraint_json(value: &serde_json::Value) -> PyResult<Box<dyn ConstraintEvaluator>> {
    let spec: ConstraintSpec = serde_json::from_value(value.clone()).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid constraint JSON: {e}"))
    })?;
    spec.into_evaluator()
}
