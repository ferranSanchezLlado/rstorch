use rstorch::shape::{DimEntry, DimId, DimSpec};
use rstorch::{Shape, ShapeSpec, StaticShape};

struct RogueDim;

impl DimSpec for RogueDim {
    fn known() -> Option<usize> {
        None
    }

    fn symbol() -> Option<DimId> {
        None
    }

    fn symbol_name() -> Option<&'static str> {
        None
    }
}

struct RogueShape;

impl ShapeSpec for RogueShape {
    const RANK: usize = 0;

    fn known_shape() -> Option<Shape> {
        Some(Shape::known([]))
    }

    fn dim_entries(_operand: usize) -> Vec<DimEntry> {
        Vec::new()
    }
}

impl StaticShape for RogueShape {
    fn static_shape() -> Shape {
        Shape::known([])
    }
}

fn main() {}
