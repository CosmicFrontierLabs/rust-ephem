//! Constraint evaluation modules
//!
//! This module provides constraint evaluation for astronomical observations,
//! including sun/moon proximity, eclipse detection, and earth limb constraints.

pub mod constraint_wrapper;
pub mod core;

// Re-export main types
pub use constraint_wrapper::PyConstraint;
pub use core::{ConstraintResult, ConstraintViolation, VisibilityWindow};
