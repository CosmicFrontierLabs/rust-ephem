//! Polygon utility functions

/// Check if a point (x, y) is inside a polygon using the winding number algorithm
/// This is more robust than ray casting for complex polygons
pub fn point_in_polygon(polygon: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut winding_number = 0.0;
    let n = polygon.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let (x1, y1) = polygon[i];
        let (x2, y2) = polygon[j];

        if y1 <= y {
            if y2 > y && (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0.0 {
                winding_number += 1.0;
            }
        } else if y2 <= y && (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0.0 {
            winding_number -= 1.0;
        }
    }

    winding_number != 0.0
}

/// Check if a point violates a polygon constraint (i.e., is outside the polygon if one is defined)
pub fn polygon_violation(polygon: Option<&Vec<(f64, f64)>>, x: f64, y: f64) -> bool {
    polygon.is_some_and(|poly| !point_in_polygon(poly, x, y))
}
