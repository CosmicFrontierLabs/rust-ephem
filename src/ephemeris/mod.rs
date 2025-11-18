//! Ephemeris computation modules
//!
//! This module contains implementations for computing celestial body positions
//! and velocities using various data sources (SPICE, TLE, ground stations, CCSDS).

pub mod ccsds_ephemeris;
pub mod ephemeris_common;
pub mod ground_ephemeris;
pub mod position_velocity;
pub mod spice_ephemeris;
pub mod spice_manager;
pub mod tle_ephemeris;

// Re-export main types
pub use ccsds_ephemeris::OEMEphemeris;
pub use ground_ephemeris::GroundEphemeris;
pub use spice_ephemeris::SPICEEphemeris;
pub use tle_ephemeris::TLEEphemeris;
