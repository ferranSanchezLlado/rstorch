//! Crate-private helpers for readable diagnostics during const evaluation.
#![allow(dead_code, unused_imports, unused_macros)]

// Fits the longest planned const-check diagnostic: operation text, labels, and
// up to three `usize::MAX` values, with more than 100 bytes of headroom.
const MESSAGE_CAPACITY: usize = 256;

pub(crate) struct ConstWriter {
    buf: [u8; MESSAGE_CAPACITY],
    len: usize,
}

impl ConstWriter {
    pub(crate) const fn new() -> Self {
        Self {
            buf: [0; MESSAGE_CAPACITY],
            len: 0,
        }
    }

    pub(crate) const fn str(mut self, value: &str) -> Self {
        let bytes = value.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            self = self.byte(bytes[index]);
            index += 1;
        }
        self
    }

    pub(crate) const fn num(mut self, value: usize) -> Self {
        if value == 0 {
            return self.byte(b'0');
        }

        let mut digits = [0_u8; 20];
        let mut digit_count = 0;
        let mut remaining = value;
        while remaining > 0 {
            digits[digit_count] = b'0' + (remaining % 10) as u8;
            remaining /= 10;
            digit_count += 1;
        }

        while digit_count > 0 {
            digit_count -= 1;
            self = self.byte(digits[digit_count]);
        }

        self
    }

    #[track_caller]
    pub(crate) const fn panic(&self) -> ! {
        match core::str::from_utf8(self.bytes()) {
            Ok(message) => panic!("{}", message),
            Err(_) => panic!("const check message was not valid UTF-8"),
        }
    }

    const fn byte(mut self, value: u8) -> Self {
        if self.len < MESSAGE_CAPACITY {
            self.buf[self.len] = value;
            self.len += 1;
        }
        self
    }

    const fn bytes(&self) -> &[u8] {
        // Slicing through `Index` is not const-stable on this nightly. `len` is
        // bounded by `MESSAGE_CAPACITY` because `byte` is the only writer.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr(), self.len) }
    }
}

#[track_caller]
pub const fn size_eq(lhs: usize, rhs: usize, label: &str, lhs_name: &str, rhs_name: &str) -> usize {
    if lhs != rhs {
        ConstWriter::new()
            .str(label)
            .str(": ")
            .str(lhs_name)
            .str("=")
            .num(lhs)
            .str(" must equal ")
            .str(rhs_name)
            .str("=")
            .num(rhs)
            .panic();
    }
    0
}

#[track_caller]
pub const fn nonzero(value: usize, label: &str, name: &str) -> usize {
    if value == 0 {
        ConstWriter::new()
            .str(label)
            .str(": ")
            .str(name)
            .str(" must be greater than 0")
            .panic();
    }
    0
}

macro_rules! const_size_eq {
    ($lhs:expr, $rhs:expr, $label:literal, $lhs_name:literal, $rhs_name:literal) => {{
        let _ = $crate::const_check::size_eq($lhs, $rhs, $label, $lhs_name, $rhs_name);
    }};
}

macro_rules! const_nonzero {
    ($value:expr, $label:literal, $name:literal) => {{
        let _ = $crate::const_check::nonzero($value, $label, $name);
    }};
}

pub(crate) use const_nonzero;
pub(crate) use const_size_eq;

#[cfg(test)]
mod tests {
    use super::{ConstWriter, const_nonzero, const_size_eq};

    fn message(writer: ConstWriter) -> String {
        core::str::from_utf8(writer.bytes()).unwrap().to_owned()
    }

    #[test]
    fn num_formats_zero() {
        assert_eq!(message(ConstWriter::new().num(0)), "0");
    }

    #[test]
    fn num_formats_single_digit() {
        assert_eq!(message(ConstWriter::new().num(7)), "7");
    }

    #[test]
    fn num_formats_multi_digit() {
        assert_eq!(message(ConstWriter::new().num(12_345)), "12345");
    }

    #[test]
    fn num_formats_large_value() {
        assert_eq!(
            message(ConstWriter::new().num(usize::MAX)),
            usize::MAX.to_string()
        );
    }

    #[test]
    fn builds_composed_message() {
        let writer = ConstWriter::new()
            .str("reshape_2d")
            .str(": source A*B*C=")
            .num(12)
            .str(" must equal target M*N=")
            .num(10);

        assert_eq!(
            message(writer),
            "reshape_2d: source A*B*C=12 must equal target M*N=10"
        );
    }

    #[test]
    fn passing_macros_compile_to_noop() {
        const _: () = const_size_eq!(12, 12, "reshape_2d", "source", "target");
        const _: () = const_nonzero!(1, "softmax_rows", "N");
    }
}
