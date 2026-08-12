/// Shared spacecraft-roll composition formula.
///
/// A "coordinated spacecraft roll" is one candidate roll shared unchanged by every
/// boresight node in a constraint tree, expressed in a single fixed physical
/// CCW-positive convention — the same one `roll_clockwise = false` uses. Each node
/// additionally has its own fixed instrument mounting angle (`mounting_roll_deg`),
/// which is expressed in that node's own `roll_clockwise` convention.
///
/// Composing the two means converting only the mounting angle through its own
/// convention and adding the candidate roll unchanged:
///
/// ```text
/// composed_ccw = (mounting_clockwise ? -mounting_roll_deg : mounting_roll_deg) + candidate_roll_deg
/// ```
///
/// Re-signing `candidate_roll_deg` per node instead (as a naive
/// `mounting_roll_deg + candidate_roll_deg` then flipped through `roll_clockwise`
/// would do) shifts the relative orientation between a CW- and a CCW-convention
/// node by twice the candidate roll as it sweeps, defeating the "coordinated
/// spacecraft roll" this composition models. This exact bug is what regressed in
/// `roll_range.rs::roll_sweep_vec` prior to this fix — the formula is now
/// centralized here so `roll_range.rs`, `boresight.rs`, and `py_api_helpers.rs`
/// cannot drift from one another again.
pub(super) fn coordinated_roll_ccw_deg(
    mounting_roll_deg: f64,
    mounting_clockwise: bool,
    candidate_roll_deg: f64,
) -> f64 {
    let mounting_ccw = if mounting_clockwise {
        -mounting_roll_deg
    } else {
        mounting_roll_deg
    };
    mounting_ccw + candidate_roll_deg
}
