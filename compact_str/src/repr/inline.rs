use core::mem::MaybeUninit;
use core::{ptr, slice};

use super::last_utf8_char::LastUtf8Char;
use super::{Repr, LENGTH_MASK, MAX_SIZE};

/// A buffer stored on the stack whose size is equal to the stack size of `String`
#[repr(C)]
pub(crate) struct InlineBuffer {
    buf: [u8; MAX_SIZE - 1],
    last_char: LastUtf8Char
}

impl InlineBuffer {
    /// Construct a new [`InlineString`]. A string that lives in a small buffer on the stack
    ///
    /// Returns `None` if the length of `text` is greater than [`MAX_SIZE`].
    #[inline]
    pub fn new(s: &str) -> Option<InlineBuffer> {
        let mut short_buf = [0u8; MAX_SIZE - 1];
        let last_char;
        if s.len() == MAX_SIZE {
            short_buf[..MAX_SIZE - 1].copy_from_slice(&s.as_bytes()[..MAX_SIZE - 1]);
            last_char = LastUtf8Char::from_utf8_byte(s.as_bytes()[MAX_SIZE - 1]);
        } else {
            // pull this out so that LLVM can optimize the match better
            if s.len() > MAX_SIZE {
                return None;
            }
            last_char = LastUtf8Char::from_len(s.len());
            short_buf[..s.len()].copy_from_slice(&s.as_bytes());
        }
        Some(InlineBuffer {
            buf: short_buf, last_char
        })
    }

    #[inline]
    pub const fn new_const(text: &str) -> Self {
        if text.len() > MAX_SIZE {
            panic!("Provided string has a length greater than our MAX_SIZE");
        }
        let mut short_buf = [0u8; MAX_SIZE - 1];
        let last_char;
        if text.len() == MAX_SIZE {
            short_buf[..MAX_SIZE - 1].copy_from_slice(&text.as_bytes()[..MAX_SIZE - 1]);
            last_char = LastUtf8Char::from_utf8_byte(text.as_bytes()[MAX_SIZE - 1]);
        } else {
            last_char = LastUtf8Char::from_len(text.len());
            short_buf[..text.len()].copy_from_slice(&text.as_bytes());
        }
        InlineBuffer {
            buf: short_buf, last_char
        }
    }

    /// Returns an empty [`InlineBuffer`]
    #[inline(always)]
    pub const fn empty() -> Self {
        const EMPTY: InlineBuffer = InlineBuffer::new_const("");
        EMPTY
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.last_char.as_len()
    }

    /// Consumes the [`InlineBuffer`] returning the entire underlying array and the length of the
    /// string that it contains
    #[inline]
    #[cfg(feature = "smallvec")]
    pub fn into_array(self) -> ([u8; MAX_SIZE], usize) {
        let mut buffer = [0; MAX_SIZE];
        buffer[..MAX_SIZE - 1].copy_from_slice(&self.buf);
        buffer[MAX_SIZE - 1] = self.last_char as u8;
        (buffer, self.len())
    }

    /// Sets the length of the content for this [`InlineBuffer`]
    ///
    /// # SAFETY:
    /// * The caller must guarantee that `len` bytes in the buffer are valid UTF-8
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= MAX_SIZE);

        // If `length` == MAX_SIZE, then we infer the length to be the capacity of the buffer. We
        // can infer this because the way we encode length doesn't overlap with any valid UTF-8
        // bytes
        if len < MAX_SIZE {
            self.last_char = LastUtf8Char::from_len(len);
        }
    }

    pub(crate) unsafe fn as_mut_buf(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: the caller must guarantee that they only write valid UTF-8 last-bytes into `last_char`
        slice::from_raw_parts_mut(self as *mut Self as *mut MaybeUninit<u8>, MAX_SIZE)
    }
}

#[cfg(test)]
mod tests {
    #[rustversion::since(1.63)]
    #[test]
    #[ignore] // we run this in CI, but unless you're compiling in release, this takes a while
    fn test_unused_utf8_bytes() {
        use rayon::prelude::*;

        // test to validate for all char the first and last bytes are never within a specified range
        // note: according to the UTF-8 spec it shouldn't be, but we double check that here
        (0..u32::MAX).into_par_iter().for_each(|i| {
            if let Ok(c) = char::try_from(i) {
                let mut buf = [0_u8; 4];
                c.encode_utf8(&mut buf);

                // check ranges for first byte
                match buf[0] {
                    x @ 128..=191 => panic!("first byte within 128..=191, {}", x),
                    x @ 248..=255 => panic!("first byte within 248..=255, {}", x),
                    _ => (),
                }

                // check ranges for last byte
                if let x @ 192..=255 = buf[c.len_utf8() - 1] {
                    panic!("last byte within 192..=255, {}", x)
                }
            }
        })
    }

    #[cfg(feature = "smallvec")]
    mod smallvec {
        use alloc::string::String;

        use quickcheck_macros::quickcheck;

        use crate::repr::{InlineBuffer, MAX_SIZE};

        #[test]
        fn test_into_array() {
            let s = "hello world!";

            let inline = unsafe { InlineBuffer::new(s) };
            let (array, length) = inline.into_array();

            assert_eq!(s.len(), length);

            // all bytes after the length should be 0
            assert!(array[length..].iter().all(|b| *b == 0));

            // taking a string slice should give back the same string as the original
            let ex_s = unsafe { core::str::from_utf8_unchecked(&array[..length]) };
            assert_eq!(s, ex_s);
        }

        #[quickcheck]
        #[cfg_attr(miri, ignore)]
        fn quickcheck_into_array(s: String) {
            let mut total_length = 0;
            let s: String = s
                .chars()
                .take_while(|c| {
                    total_length += c.len_utf8();
                    total_length < MAX_SIZE
                })
                .collect();

            let inline = unsafe { InlineBuffer::new(&s) };
            let (array, length) = inline.into_array();
            assert_eq!(s.len(), length);

            // all bytes after the length should be 0
            assert!(array[length..].iter().all(|b| *b == 0));

            // taking a string slice should give back the same string as the original
            let ex_s = unsafe { core::str::from_utf8_unchecked(&array[..length]) };
            assert_eq!(s, ex_s);
        }
    }
}
