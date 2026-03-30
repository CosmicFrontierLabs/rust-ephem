// Split from previous monolithic constraint_wrapper.rs into focused sections.
// Include order is important: later sections depend on earlier symbols.

mod py_api;
pub use py_api::*;

mod field_of_regard;

mod roll_range;

mod json_parser;

mod combinators;

mod boresight;

mod json_to_py;
