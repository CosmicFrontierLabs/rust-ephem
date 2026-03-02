// Split from previous monolithic constraint_wrapper.rs into focused sections.
// Include order is important: later sections depend on earlier symbols.

include!("py_api.rs");
include!("json_parser.rs");
include!("combinators.rs");
include!("boresight.rs");
include!("json_to_py.rs");
