use smallvec::SmallVec;

use super::{Repr, MAX_SIZE};

impl Repr {
    /// Consumes the [`Repr`] returning a byte vector in a [`SmallVec`]
    ///
    /// Note: both for the inlined case and the heap case, the buffers are re-used
    #[inline]
    pub fn into_bytes(self) -> SmallVec<[u8; MAX_SIZE]> {
        match self {
            Repr::Inline(inline) => {
                // SAFETY: We just checked the discriminant to make sure we're an InlineBuffer
                let (array, length) = inline.into_array();
                SmallVec::from_buf_and_len(array, length)
            }
            Repr::Heap(heap) => SmallVec::from_vec(heap.into_string().into_bytes()),
            Repr::ThinHeap(thin_heap) => SmallVec::from_vec(thin_heap.into_string().into_bytes()),
            Repr::Static(s) => SmallVec::from(s.as_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use crate::CompactString;

    #[test_case("" ; "empty")]
    #[test_case("abc" ; "short")]
    #[test_case("I am a long string 😊😊😊😊😊" ; "long")]
    fn proptest_roundtrip(s: &'static str) {
        let og_compact = CompactString::from(s);
        assert_eq!(og_compact, s);

        let bytes = og_compact.into_bytes();

        let ex_compact = CompactString::from_utf8(bytes).unwrap();
        assert_eq!(ex_compact, s);

        // test `StaticStr` variant
        let og_compact = CompactString::from_static_str(s);
        assert_eq!(og_compact, s);

        let bytes = og_compact.into_bytes();

        let ex_compact = CompactString::from_utf8(bytes).unwrap();
        assert_eq!(ex_compact, s);
    }
}
