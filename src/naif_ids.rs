/// NAIF ID definitions and name mappings for solar system bodies
///
/// This module provides mappings between common body names and their NAIF IDs,
/// allowing bodies to be specified either by ID or by name (case-insensitive).
///
/// Reference: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
use once_cell::sync::Lazy;
use std::collections::HashMap;

// Solar System Barycenter and Sun
pub const SOLAR_SYSTEM_BARYCENTER: i32 = 0;
pub const SUN: i32 = 10;

// Planets
pub const MERCURY: i32 = 199;
pub const VENUS: i32 = 299;
pub const EARTH: i32 = 399;
pub const MARS: i32 = 499;
pub const JUPITER: i32 = 599;
pub const SATURN: i32 = 699;
pub const URANUS: i32 = 799;
pub const NEPTUNE: i32 = 899;
pub const PLUTO: i32 = 999;

// Planet barycenters
pub const MERCURY_BARYCENTER: i32 = 1;
pub const VENUS_BARYCENTER: i32 = 2;
pub const EARTH_BARYCENTER: i32 = 3;
pub const MARS_BARYCENTER: i32 = 4;
pub const JUPITER_BARYCENTER: i32 = 5;
pub const SATURN_BARYCENTER: i32 = 6;
pub const URANUS_BARYCENTER: i32 = 7;
pub const NEPTUNE_BARYCENTER: i32 = 8;
pub const PLUTO_BARYCENTER: i32 = 9;

// Earth's Moon
pub const MOON: i32 = 301;

// Mars moons
pub const PHOBOS: i32 = 401;
pub const DEIMOS: i32 = 402;

// Jupiter major moons
pub const IO: i32 = 501;
pub const EUROPA: i32 = 502;
pub const GANYMEDE: i32 = 503;
pub const CALLISTO: i32 = 504;

// Saturn major moons
pub const MIMAS: i32 = 601;
pub const ENCELADUS: i32 = 602;
pub const TETHYS: i32 = 603;
pub const DIONE: i32 = 604;
pub const RHEA: i32 = 605;
pub const TITAN: i32 = 606;
pub const HYPERION: i32 = 607;
pub const IAPETUS: i32 = 608;

// Uranus major moons
pub const ARIEL: i32 = 701;
pub const UMBRIEL: i32 = 702;
pub const TITANIA: i32 = 703;
pub const OBERON: i32 = 704;
pub const MIRANDA: i32 = 705;

// Neptune major moons
pub const TRITON: i32 = 801;

/// Global hashmap for name -> NAIF ID lookups (case-insensitive)
pub static BODY_NAME_TO_ID: Lazy<HashMap<String, i32>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Solar System Barycenter and Sun
    map.insert("ssb".to_lowercase(), SOLAR_SYSTEM_BARYCENTER);
    map.insert(
        "solar system barycenter".to_lowercase(),
        SOLAR_SYSTEM_BARYCENTER,
    );
    map.insert("sun".to_lowercase(), SUN);

    // Planets
    map.insert("mercury".to_lowercase(), MERCURY);
    map.insert("venus".to_lowercase(), VENUS);
    map.insert("earth".to_lowercase(), EARTH);
    map.insert("mars".to_lowercase(), MARS);
    map.insert("jupiter".to_lowercase(), JUPITER);
    map.insert("saturn".to_lowercase(), SATURN);
    map.insert("uranus".to_lowercase(), URANUS);
    map.insert("neptune".to_lowercase(), NEPTUNE);
    map.insert("pluto".to_lowercase(), PLUTO);

    // Planet barycenters
    map.insert("mercury barycenter".to_lowercase(), MERCURY_BARYCENTER);
    map.insert("venus barycenter".to_lowercase(), VENUS_BARYCENTER);
    map.insert("earth barycenter".to_lowercase(), EARTH_BARYCENTER);
    map.insert("mars barycenter".to_lowercase(), MARS_BARYCENTER);
    map.insert("jupiter barycenter".to_lowercase(), JUPITER_BARYCENTER);
    map.insert("saturn barycenter".to_lowercase(), SATURN_BARYCENTER);
    map.insert("uranus barycenter".to_lowercase(), URANUS_BARYCENTER);
    map.insert("neptune barycenter".to_lowercase(), NEPTUNE_BARYCENTER);
    map.insert("pluto barycenter".to_lowercase(), PLUTO_BARYCENTER);

    // Earth's Moon
    map.insert("moon".to_lowercase(), MOON);
    map.insert("luna".to_lowercase(), MOON);

    // Mars moons
    map.insert("phobos".to_lowercase(), PHOBOS);
    map.insert("deimos".to_lowercase(), DEIMOS);

    // Jupiter major moons
    map.insert("io".to_lowercase(), IO);
    map.insert("europa".to_lowercase(), EUROPA);
    map.insert("ganymede".to_lowercase(), GANYMEDE);
    map.insert("callisto".to_lowercase(), CALLISTO);

    // Saturn major moons
    map.insert("mimas".to_lowercase(), MIMAS);
    map.insert("enceladus".to_lowercase(), ENCELADUS);
    map.insert("tethys".to_lowercase(), TETHYS);
    map.insert("dione".to_lowercase(), DIONE);
    map.insert("rhea".to_lowercase(), RHEA);
    map.insert("titan".to_lowercase(), TITAN);
    map.insert("hyperion".to_lowercase(), HYPERION);
    map.insert("iapetus".to_lowercase(), IAPETUS);

    // Uranus major moons
    map.insert("ariel".to_lowercase(), ARIEL);
    map.insert("umbriel".to_lowercase(), UMBRIEL);
    map.insert("titania".to_lowercase(), TITANIA);
    map.insert("oberon".to_lowercase(), OBERON);
    map.insert("miranda".to_lowercase(), MIRANDA);

    // Neptune major moons
    map.insert("triton".to_lowercase(), TRITON);

    map
});

/// Convert a body name (case-insensitive) to its NAIF ID
///
/// # Arguments
/// * `name` - Body name (e.g., "Jupiter", "Mars", "moon")
///
/// # Returns
/// `Some(naif_id)` if the name is recognized, `None` otherwise
///
/// # Example
/// ```
/// use rust_ephem::naif_ids::name_to_id;
/// assert_eq!(name_to_id("Jupiter"), Some(599));
/// assert_eq!(name_to_id("moon"), Some(301));
/// assert_eq!(name_to_id("EARTH"), Some(399));
/// ```
pub fn name_to_id(name: &str) -> Option<i32> {
    BODY_NAME_TO_ID.get(&name.to_lowercase()).copied()
}

/// Parse a body identifier that can be either a NAIF ID (integer) or a body name (string)
///
/// # Arguments
/// * `identifier` - Either a NAIF ID as string (e.g., "599") or body name (e.g., "Jupiter")
///
/// # Returns
/// `Some(naif_id)` if the identifier is valid, `None` otherwise
///
/// # Example
/// ```
/// use rust_ephem::naif_ids::parse_body_identifier;
/// assert_eq!(parse_body_identifier("599"), Some(599));
/// assert_eq!(parse_body_identifier("Jupiter"), Some(599));
/// assert_eq!(parse_body_identifier("moon"), Some(301));
/// ```
pub fn parse_body_identifier(identifier: &str) -> Option<i32> {
    // Try to parse as integer first
    if let Ok(id) = identifier.parse::<i32>() {
        return Some(id);
    }

    // Otherwise try to look up as a name
    name_to_id(identifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_to_id() {
        assert_eq!(name_to_id("sun"), Some(SUN));
        assert_eq!(name_to_id("Sun"), Some(SUN));
        assert_eq!(name_to_id("SUN"), Some(SUN));
        assert_eq!(name_to_id("earth"), Some(EARTH));
        assert_eq!(name_to_id("moon"), Some(MOON));
        assert_eq!(name_to_id("luna"), Some(MOON));
        assert_eq!(name_to_id("jupiter"), Some(JUPITER));
        assert_eq!(name_to_id("io"), Some(IO));
        assert_eq!(name_to_id("unknown"), None);
    }

    #[test]
    fn test_parse_body_identifier() {
        // Test with NAIF IDs as strings
        assert_eq!(parse_body_identifier("10"), Some(SUN));
        assert_eq!(parse_body_identifier("399"), Some(EARTH));
        assert_eq!(parse_body_identifier("301"), Some(MOON));

        // Test with names
        assert_eq!(parse_body_identifier("Sun"), Some(SUN));
        assert_eq!(parse_body_identifier("earth"), Some(EARTH));
        assert_eq!(parse_body_identifier("MOON"), Some(MOON));

        // Test invalid
        assert_eq!(parse_body_identifier("invalid"), None);
    }
}
