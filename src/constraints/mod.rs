//! Constraint evaluation modules
//!
//! This module provides constraint evaluation for astronomical observations,
//! including sun/moon proximity, eclipse detection, and earth limb constraints.

// Macro must be defined before modules that use it
#[macro_use]
pub mod core;

// Constraint implementations
pub mod body_proximity;
pub mod earth_limb;
pub mod eclipse;
pub mod moon_proximity;
pub mod sun_proximity;

// Python wrapper
pub mod constraint_wrapper;

// Re-export main types for public API
pub use constraint_wrapper::PyConstraint;
pub use core::{ConstraintResult, ConstraintViolation, VisibilityWindow};
