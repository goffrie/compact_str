use static_assertions::*;
use std::mem::ManuallyDrop;

mod discriminant;
mod heap;
mod packed;
mod inline;

use discriminant::{
    Discriminant,
    DiscriminantMask,
};
use heap::HeapString;
use inline::InlineString;
use packed::PackedString;

const MAX_SIZE: usize = std::mem::size_of::<String>();

pub union Repr {
    mask: DiscriminantMask,
    heap: ManuallyDrop<HeapString>,
    inline: InlineString,
    packed: PackedString,
}

impl Repr {
    #[inline]
    pub fn new<T: AsRef<str>>(text: T) -> Self {
        let text = text.as_ref();
        let len = text.len();

        if len <= inline::MAX_INLINE_SIZE {
            let inline = InlineString::new(text);
            Repr { inline }
        } else if len == MAX_SIZE && text.as_bytes()[0] <= 127 {
            let packed = PackedString::new(text);
            Repr { packed }
        } else {
            let heap = ManuallyDrop::new(HeapString::new(text));
            Repr { heap }
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        self.cast().into_str()
    }

    #[inline]
    pub fn is_heap_allocated(&self) -> bool {
        matches!(self.cast(), StrongRepr::Heap(..))
    }

    #[inline]
    fn discriminant(&self) -> Discriminant {
        // SAFETY: `heap`, `inline`, and `packed` all store a discriminant in their first byte
        unsafe { self.mask.disciminant() }
    }

    #[inline]
    fn cast(&self) -> StrongRepr<'_> {
        match self.discriminant() {
            Discriminant::Heap => {
                // SAFETY: We checked the discriminant to make sure the union is `heap`
                StrongRepr::Heap(unsafe { &self.heap })
            }
            Discriminant::Inline => {
                // SAFETY: We checked the discriminant to make sure the union is `inline`
                StrongRepr::Inline(unsafe { &self.inline })
            }
            Discriminant::Packed => {
                // SAFETY: We checked the discriminant to make sure the union is `packed`
                StrongRepr::Packed(unsafe { &self.packed })
            }
        }
    }
}

impl Clone for Repr {
    fn clone(&self) -> Self {
        match self.cast() {
            StrongRepr::Heap(heap) => Repr { heap: heap.clone() },
            StrongRepr::Inline(inline) => Repr { inline: *inline },
            StrongRepr::Packed(packed) => Repr { packed: *packed },
        }
    }
}

impl Drop for Repr {
    fn drop(&mut self) {
        match self.discriminant() {
            Discriminant::Heap => {
                // SAFETY: We checked the discriminant to make sure the union is `heap`
                unsafe { ManuallyDrop::drop(&mut self.heap) };
            }
            // No-op, the value is on the stack and doesn't need to be explicitly dropped
            Discriminant::Inline | Discriminant::Packed => {}
        }
    }
}

#[derive(Debug)]
enum StrongRepr<'a> {
    Heap(&'a ManuallyDrop<HeapString>),
    Inline(&'a InlineString),
    Packed(&'a PackedString),
}

impl<'a> StrongRepr<'a> {
    #[inline]
    pub fn into_str(self) -> &'a str {
        match self {
            Self::Inline(inline) => {
                let len = inline.metadata.data() as usize;
                let slice = &inline.buffer[..len];

                // SAFETY: You can only construct an InlineString via a &str
                unsafe { ::std::str::from_utf8_unchecked(slice) }
            }
            Self::Packed(packed) => {
                // SAFETY: You can only construct a PackedString via a &str
                unsafe { ::std::str::from_utf8_unchecked(&packed.buffer) }
            }
            Self::Heap(heap) => &*heap.string,
        }
    }
}

assert_eq_size!(Repr, String);

#[cfg(target_pointer_width = "64")]
const_assert_eq!(std::mem::size_of::<Repr>(), 24);

#[cfg(target_pointer_width = "32")]
const_assert_eq!(std::mem::size_of::<Repr>(), 12);

#[cfg(test)]
mod tests {
    use super::Repr;

    #[test]
    fn test_inline_str() {
        let short = "abc";
        let repr = Repr::new(&short);
        assert_eq!(repr.as_str(), short);
    }

    #[test]
    fn test_packed_str() {
        #[cfg(target_pointer_width = "64")]
        let packed = "this string is 24 chars!";
        #[cfg(target_pointer_width = "32")]
        let packed = "i am 12 char";

        let repr = Repr::new(&packed);
        assert_eq!(repr.as_str(), packed);
    }

    #[test]
    fn test_heap_str() {
        let long = "I am a long string that has very many characters";
        let repr = Repr::new(&long);
        assert_eq!(repr.as_str(), long);
    }
}
