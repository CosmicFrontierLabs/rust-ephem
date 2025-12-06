//! Leap second utilities using hifitime
//!
//! TT-UTC = TT-TAI + TAI-UTC = 32.184 + leap_seconds

pub use crate::utils::hifi_time::get_tai_utc_offset;
