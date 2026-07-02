const BUFFER_LEN: usize = 512;
const ELLIPSIS_LEN: usize = 3;

/// Small const-eval string builder for readable compile-time diagnostics.
///
/// Current messages fit well below 512 bytes even with four `usize::MAX` values.
/// If future labels exceed that budget, the writer truncates with `...` instead
/// of failing with a secondary buffer-overflow panic.
pub(crate) struct ConstWriter {
    buf: [u8; BUFFER_LEN],
    len: usize,
    truncated: bool,
}

impl ConstWriter {
    pub(crate) const fn new() -> Self {
        Self {
            buf: [0; BUFFER_LEN],
            len: 0,
            truncated: false,
        }
    }

    pub(crate) const fn str(mut self, value: &str) -> Self {
        let bytes = value.as_bytes();
        let mut idx = 0usize;
        while idx < bytes.len() {
            self = self.byte(bytes[idx]);
            idx += 1;
        }
        self
    }

    pub(crate) const fn num(mut self, mut value: usize) -> Self {
        if value == 0 {
            return self.byte(b'0');
        }

        let mut digits = [0u8; 20];
        let mut len = 0usize;
        while value > 0 {
            digits[len] = b'0' + (value % 10) as u8;
            value /= 10;
            len += 1;
        }

        while len > 0 {
            len -= 1;
            self = self.byte(digits[len]);
        }
        self
    }

    #[track_caller]
    pub(crate) const fn panic(&self) -> ! {
        panic!("{}", self.as_str())
    }

    pub(crate) const fn as_str(&self) -> &str {
        // The writer only appends ASCII bytes and keeps len within BUFFER_LEN.
        let bytes = unsafe { core::slice::from_raw_parts(self.buf.as_ptr(), self.len) };
        match core::str::from_utf8(bytes) {
            Ok(value) => value,
            Err(_) => "const check failed",
        }
    }

    const fn byte(mut self, value: u8) -> Self {
        if self.truncated {
            return self;
        }

        if self.len < BUFFER_LEN - ELLIPSIS_LEN {
            self.buf[self.len] = value;
            self.len += 1;
            return self;
        }

        self.buf[self.len] = b'.';
        self.buf[self.len + 1] = b'.';
        self.buf[self.len + 2] = b'.';
        self.len += ELLIPSIS_LEN;
        self.truncated = true;
        self
    }
}

#[track_caller]
pub(crate) const fn size_eq(
    lhs: usize,
    rhs: usize,
    op: &str,
    lhs_name: &str,
    rhs_name: &str,
) -> usize {
    if lhs != rhs {
        ConstWriter::new()
            .str(op)
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
pub(crate) const fn known_size_eq(
    lhs: Option<usize>,
    rhs: Option<usize>,
    op: &str,
    lhs_name: &str,
    rhs_name: &str,
) -> usize {
    if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
        size_eq(lhs, rhs, op, lhs_name, rhs_name)
    } else {
        0
    }
}

#[track_caller]
pub(crate) const fn sum_eq(
    lhs: usize,
    rhs: usize,
    out: usize,
    op: &str,
    lhs_name: &str,
    rhs_name: &str,
    out_name: &str,
) -> usize {
    match lhs.checked_add(rhs) {
        Some(sum) => {
            if sum != out {
                ConstWriter::new()
                    .str(op)
                    .str(": ")
                    .str(lhs_name)
                    .str("=")
                    .num(lhs)
                    .str(" plus ")
                    .str(rhs_name)
                    .str("=")
                    .num(rhs)
                    .str(" gives ")
                    .num(sum)
                    .str(", but ")
                    .str(out_name)
                    .str("=")
                    .num(out)
                    .panic();
            }
        }
        None => ConstWriter::new()
            .str(op)
            .str(": ")
            .str(lhs_name)
            .str("=")
            .num(lhs)
            .str(" plus ")
            .str(rhs_name)
            .str("=")
            .num(rhs)
            .str(" overflows usize")
            .panic(),
    }
    0
}

#[track_caller]
pub(crate) const fn known_sum_eq(
    lhs: Option<usize>,
    rhs: Option<usize>,
    out: usize,
    op: &str,
    lhs_name: &str,
    rhs_name: &str,
    out_name: &str,
) -> usize {
    if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
        sum_eq(lhs, rhs, out, op, lhs_name, rhs_name, out_name)
    } else {
        0
    }
}

#[track_caller]
pub(crate) const fn sum_nonzero(
    lhs: usize,
    rhs: usize,
    op: &str,
    lhs_name: &str,
    rhs_name: &str,
    sum_name: &str,
) -> usize {
    match lhs.checked_add(rhs) {
        Some(sum) => {
            if sum == 0 {
                ConstWriter::new()
                    .str(op)
                    .str(": ")
                    .str(lhs_name)
                    .str("=")
                    .num(lhs)
                    .str(" plus ")
                    .str(rhs_name)
                    .str("=")
                    .num(rhs)
                    .str(" gives ")
                    .str(sum_name)
                    .str("=0, which must be greater than 0")
                    .panic();
            }
        }
        None => ConstWriter::new()
            .str(op)
            .str(": ")
            .str(lhs_name)
            .str("=")
            .num(lhs)
            .str(" plus ")
            .str(rhs_name)
            .str("=")
            .num(rhs)
            .str(" overflows usize")
            .panic(),
    }
    0
}

#[track_caller]
pub(crate) const fn mul_eq(
    lhs: usize,
    rhs: usize,
    out: usize,
    op: &str,
    lhs_name: &str,
    rhs_name: &str,
    out_name: &str,
) -> usize {
    match lhs.checked_mul(rhs) {
        Some(product) => {
            if product != out {
                ConstWriter::new()
                    .str(op)
                    .str(": ")
                    .str(lhs_name)
                    .str("=")
                    .num(lhs)
                    .str(" times ")
                    .str(rhs_name)
                    .str("=")
                    .num(rhs)
                    .str(" gives ")
                    .num(product)
                    .str(", but ")
                    .str(out_name)
                    .str("=")
                    .num(out)
                    .panic();
            }
        }
        None => ConstWriter::new()
            .str(op)
            .str(": ")
            .str(lhs_name)
            .str("=")
            .num(lhs)
            .str(" times ")
            .str(rhs_name)
            .str("=")
            .num(rhs)
            .str(" overflows usize")
            .panic(),
    }
    0
}

#[track_caller]
pub(crate) const fn mul_fits(
    lhs: usize,
    rhs: usize,
    op: &str,
    lhs_name: &str,
    rhs_name: &str,
) -> usize {
    match lhs.checked_mul(rhs) {
        Some(product) => product,
        None => ConstWriter::new()
            .str(op)
            .str(": ")
            .str(lhs_name)
            .str("=")
            .num(lhs)
            .str(" times ")
            .str(rhs_name)
            .str("=")
            .num(rhs)
            .str(" overflows usize")
            .panic(),
    }
}

#[track_caller]
pub(crate) const fn nonzero(value: usize, op: &str, name: &str) -> usize {
    if value == 0 {
        ConstWriter::new()
            .str(op)
            .str(": ")
            .str(name)
            .str("=0 must be greater than 0")
            .panic();
    }
    0
}

#[track_caller]
pub(crate) const fn known_nonzero(value: Option<usize>, op: &str, name: &str) -> usize {
    if let Some(value) = value {
        nonzero(value, op, name)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{self as const_check, ConstWriter};

    #[test]
    fn const_writer_formats_strings_and_usize_values() {
        let message = ConstWriter::new()
            .str("zero=")
            .num(0)
            .str(", small=")
            .num(7)
            .str(", large=")
            .num(123_456_789);

        assert_eq!(message.as_str(), "zero=0, small=7, large=123456789");
    }

    #[test]
    fn const_writer_truncates_instead_of_overflowing() {
        let long = "a".repeat(super::BUFFER_LEN * 2);
        let message = ConstWriter::new().str(&long);

        assert_eq!(message.as_str().len(), super::BUFFER_LEN);
        assert!(message.as_str().ends_with("..."));
    }

    #[test]
    fn const_checks_are_noops_when_conditions_hold() {
        const _: usize = const_check::size_eq(12, 12, "reshape", "source", "target");
        const _: usize =
            const_check::known_size_eq(Some(12), Some(12), "reshape", "source", "target");
        const _: usize = const_check::known_size_eq(None, Some(12), "reshape", "source", "target");
        const _: usize = const_check::sum_eq(2, 3, 5, "cat1", "lhs", "rhs", "out");
        const _: usize =
            const_check::known_sum_eq(Some(2), Some(3), 5, "cat1", "lhs", "rhs", "out");
        const _: usize = const_check::sum_nonzero(1, 0, "xavier", "in", "out", "sum");
        const _: usize = const_check::mul_eq(2, 4, 8, "heads", "heads", "dim", "embed");
        const _: usize = const_check::mul_fits(2, 4, "init", "lhs", "rhs");
        const _: usize = const_check::nonzero(1, "softmax", "axis");
        const _: usize = const_check::known_nonzero(Some(1), "softmax", "axis");
        const _: usize = const_check::known_nonzero(None, "softmax", "axis");
    }
}
