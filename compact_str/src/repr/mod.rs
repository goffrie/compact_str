use alloc::borrow::Cow;
use alloc::boxed::Box;
use core::mem::MaybeUninit;
use core::str::Utf8Error;
use core::{
    mem,
    ptr,
};

#[cfg(feature = "bytes")]
mod bytes;
#[cfg(feature = "smallvec")]
mod smallvec;

//mod capacity;
mod heap;
mod inline;
mod iter;
mod last_utf8_char;
mod num;
//mod static_str;
mod traits;

use alloc::string::String;

// use self::capacity::Capacity;
use self::heap::{HeapBuffer, ThinHeapBuffer, TooLong};
use self::inline::InlineBuffer;
use self::last_utf8_char::LastUtf8Char;
pub(crate) use traits::IntoRepr;

/// The max size of a string we can fit inline
pub const MAX_SIZE: usize = core::mem::size_of::<String>();

const EMPTY: Repr = Repr::new_inline("");

pub(crate) enum Repr {
    Inline(InlineBuffer),
    Heap(HeapBuffer),
    ThinHeap(ThinHeapBuffer),
    Static(&'static str),
}
static_assertions::assert_eq_size!([u8; MAX_SIZE], Repr);

unsafe impl Send for Repr {}
unsafe impl Sync for Repr {}

impl Repr {
    #[inline]
    pub fn new(text: &str) -> Self {
        if let Some(inline) = InlineBuffer::new(text) {
            Repr::Inline(inline)
        } else if let Ok(heap) = HeapBuffer::new(text) {
            Repr::Heap(heap)
        } else {
            Repr::ThinHeap(ThinHeapBuffer::new(text))
        }
    }

    #[inline]
    pub const fn new_inline(text: &str) -> Self {
        let len = text.len();

        if len <= MAX_SIZE {
            let inline = InlineBuffer::new_const(text);
            Repr::Inline(inline)
        } else {
            panic!("Inline string was too long, max length is `core::mem::size_of::<CompactString>()` bytes");
        }
    }

    #[inline]
    pub const fn from_static_str(text: &'static str) -> Self {
        Repr::Static(text)
    }

    /// Create a [`Repr`] with the provided `capacity`
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= MAX_SIZE {
            EMPTY
        } else if let Ok(heap) = HeapBuffer::with_capacity(capacity) {
            Repr::Heap(heap)
        } else {
            Repr::ThinHeap(ThinHeapBuffer::with_capacity(capacity))
        }
    }

    /// Create a [`Repr`] from a slice of bytes that is UTF-8
    #[inline]
    pub fn from_utf8<B: AsRef<[u8]>>(buf: B) -> Result<Self, Utf8Error> {
        // Get a &str from the Vec, failing if it's not valid UTF-8
        let s = core::str::from_utf8(buf.as_ref())?;
        // Construct a Repr from the &str
        Ok(Self::new(s))
    }

    /// Create a [`Repr`] from a slice of bytes that is UTF-8, without validating that it is indeed
    /// UTF-8
    ///
    /// # Safety
    /// * The caller must guarantee that `buf` is valid UTF-8
    #[inline]
    pub unsafe fn from_utf8_unchecked<B: AsRef<[u8]>>(buf: B) -> Self {
        let bytes = buf.as_ref();
        let bytes_len = bytes.len();

        // Create a Repr with enough capacity for the entire buffer
        let mut repr = Repr::with_capacity(bytes_len);

        // There's an edge case where the final byte of this buffer == `HEAP_MASK`, which is
        // invalid UTF-8, but would result in us creating an inline variant, that identifies as
        // a heap variant. If a user ever tried to reference the data at all, we'd incorrectly
        // try and read data from an invalid memory address, causing undefined behavior.
        if bytes_len == MAX_SIZE {
            let last_byte = bytes[bytes_len - 1];
            // If we hit the edge case, reserve additional space to make the repr becomes heap
            // allocated, which prevents us from writing this last byte inline
            if last_byte >= 0b11000000 {
                repr.reserve(MAX_SIZE + 1);
            }
        }

        // SAFETY: The caller is responsible for making sure the provided buffer is UTF-8. This
        // invariant is documented in the public API
        let slice = repr.as_mut_buf();
        // write the chunk into the Repr
        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
        slice[..bytes_len].copy_from_slice(mem::transmute::<&[u8], &[MaybeUninit<u8>]>(bytes));

        // Set the length of the Repr
        // SAFETY: We just wrote the entire `buf` into the Repr
        repr.set_len(bytes_len);

        repr
    }

    /// Create a [`Repr`] from a [`String`], in `O(1)` time. We'll attempt to inline the string
    /// if `should_inline` is `true`
    ///
    /// Note: If the provided [`String`] has a capacity >16 MB and we're on a 32-bit arch, we'll copy the
    /// `String`.
    #[inline]
    pub fn from_string(s: String, should_inline: bool) -> Self {
        if should_inline || s.capacity() == 0 {
            if let Some(inline) = InlineBuffer::new(&s) {
                return Repr::Inline(inline);
            }
        }
        match HeapBuffer::from_string(s) {
            Ok(heap) => Repr::Heap(heap),
            Err(s) => {
                // This will copy the string.
                Repr::ThinHeap(ThinHeapBuffer::new(&s))
            }
        }
    }

    /// Converts a [`Repr`] into a [`String`], in `O(1)` time, if possible
    #[inline]
    pub fn into_string(self) -> String {
        if let Repr::Heap(heap) = self {
            return heap.into_string();
        }
        self.as_str().into()
    }

    /// Reserves at least `additional` bytes. If there is already enough capacity to store
    /// `additional` bytes this is a no-op
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let len = self.len();
        let needed_capacity = len
            .checked_add(additional)
            .expect("Attempted to reserve more than 'usize::MAX' bytes");

        if !self.is_static_str() && needed_capacity <= self.capacity() {
            // we already have enough space, no-op
            // If self.is_static_str() is true, then we would have to convert
            // it to other variants since static_str variant cannot be modified.
            return;
        }

        if needed_capacity <= MAX_SIZE {
            // It's possible to have a `Repr` that is heap allocated with a capacity less than
            // MAX_SIZE, if that `Repr` was created From a String or Box<str>
            //
            // SAFTEY: Our needed_capacity is >= our length, which is <= than MAX_SIZE
            let inline = InlineBuffer::new(self.as_str()).expect("self.len() <= needed_capacity <= MAX_SIZE");
            *self = Repr::Inline(inline);
        } else {
            match self {
                Repr::Inline(_) | Repr::Static(_) => {
                    // We're not heap allocated, but need to be create a HeapBuffer
                    if let Ok(heap) = HeapBuffer::with_additional(self.as_str(), additional) {
                        *self = Repr::Heap(heap);
                    } else {
                        *self = Repr::ThinHeap(ThinHeapBuffer::with_additional(self.as_str(), additional));
                    }
                }
                Repr::Heap(heap) => {
                    let amortized_capacity = heap::amortized_growth(len, additional);
                    if let Err(TooLong) = heap.realloc(amortized_capacity) {
                        *self = Repr::ThinHeap(ThinHeapBuffer::with_additional(self.as_str(), additional));
                    }
                }
                Repr::ThinHeap(thin_heap) => {
                    let amortized_capacity = heap::amortized_growth(len, additional);
                    thin_heap.realloc(amortized_capacity)
                }
            }
        }
    }

    pub fn shrink_to(&mut self, min_capacity: usize) {
        // Note: We can't shrink the inline variant since it's buffer is a fixed size
        // or the static str variant since it is just a pointer, so we only
        // take action here if our string is heap allocated
        if let Repr::Inline(_) | Repr::Static(_) = self {
            return
        }
        let new_capacity = self.len().max(min_capacity);
        if new_capacity >= self.capacity() {
            return;
        }
        if new_capacity <= MAX_SIZE {
            // String can be inlined.
            let inline = InlineBuffer::new(self.as_str()).expect("self.len() <= new_capacity <= MAX_SIZE");
            *self = Repr::Inline(inline);
            return;
        }
        match self {
            Repr::Inline(_) | Repr::Static(_) => (),
            Repr::Heap(heap) => {
                heap.realloc(new_capacity).ok().expect("shrinking cannot fail");
            }
            Repr::ThinHeap(thin_heap) => {
                thin_heap.realloc(new_capacity);
            }
        }
    }

    #[inline]
    pub fn push_str(&mut self, s: &str) {
        // If `s` is empty, then there's no reason to reserve or push anything
        // at all.
        if s.is_empty() {
            return;
        }

        let len = self.len();
        let str_len = s.len();

        // Reserve at least enough space to fit `s`
        self.reserve(str_len);

        // SAFTEY: `s` which we're appending to the buffer, is valid UTF-8
        let slice = unsafe { self.as_mut_buf() };
        let push_buffer = &mut slice[len..len + str_len];

        debug_assert_eq!(push_buffer.len(), s.as_bytes().len());

        // Copy the string into our buffer
        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
        push_buffer.copy_from_slice(unsafe { mem::transmute::<&[u8], &[MaybeUninit<u8>]>(s.as_bytes()) });

        // Increment the length of our string
        //
        // SAFETY: We appened `s` which is valid UTF-8, and if our size became greater than
        // MAX_SIZE, our call to reserve would make us heap allocated
        unsafe { self.set_len(len + str_len) };
    }

    #[inline]
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.as_str().chars().next_back()?;

        // SAFETY: We know this is is a valid length which falls on a char boundary
        unsafe { self.set_len(self.len() - ch.len_utf8()) };

        Some(ch)
    }

    /// Returns the string content, and only the string content, as a slice of bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Repr::Inline(inline) => &inline.as_buf()[..inline.len()],
            Repr::Heap(heap) => heap.as_slice(),
            Repr::ThinHeap(thin_heap) => thin_heap.as_slice(),
            Repr::Static(s) => s.as_bytes(),
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: A `Repr` contains valid UTF-8
        unsafe { core::str::from_utf8_unchecked(self.as_slice()) }
    }

    /// Returns the length of the string that we're storing
    #[allow(clippy::len_without_is_empty)] // is_empty exists on CompactString
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Repr::Inline(inline) => inline.len(),
            Repr::Heap(heap) => heap.len(),
            Repr::ThinHeap(thin_heap) => thin_heap.len(),
            Repr::Static(s) => s.len(),
        }
    }

    /// Returns the overall capacity of the underlying buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        match self {
            Repr::Inline(_) => MAX_SIZE,
            Repr::Heap(heap) => heap.capacity(),
            Repr::ThinHeap(thin_heap) => thin_heap.capacity(),
            Repr::Static(s) => s.len(),
        }
    }

    #[inline(always)]
    pub fn is_heap_allocated(&self) -> bool {
        match self {
            Repr::Inline(_) | Repr::Static(_) => false,
            Repr::Heap(_) | Repr::ThinHeap(_) => true,
        }
    }

    #[inline(always)]
    const fn is_static_str(&self) -> bool {
        matches!(self, Repr::Static(_))
    }

    #[inline]
    #[rustversion::attr(since(1.64), const)]
    pub fn as_static_str(&self) -> Option<&'static str> {
        if let Repr::Static(s) = self {
            Some(*s)
        } else {
            None
        }
    }

    /// Return a mutable reference to the entire underlying buffer, including spare capacity
    ///
    /// # Safety
    /// * Callers can only assume that the first `self.len()` bytes of the returned buffer are initialized
    /// * Callers must only write initialized bytes into the buffer
    /// * Callers must guarantee that any modifications made to the buffer are valid UTF-8
    pub unsafe fn as_mut_buf(&mut self) -> &mut [MaybeUninit<u8>] {
        if let Some(s) = self.as_static_str() {
            *self = Repr::new(s);
        }

        match self {
            Repr::Static(_) => unreachable!(),
            Repr::Inline(inline) => {
                // SAFETY: [u8] and [MaybeUninit<u8>] have the same layout,
                // and the caller promises not to write uninitialized bytes
                mem::transmute::<&mut [u8], &mut [MaybeUninit<u8>]>(inline.as_mut_buf())
            },
            Repr::Heap(heap) => heap.as_mut_buf(),
            Repr::ThinHeap(thin_heap) => thin_heap.as_mut_buf(),
        }
    }

    /// Sets the length of the string that our underlying buffer contains
    ///
    /// # Safety
    /// * `len` bytes in the buffer must be valid UTF-8
    /// * If the underlying buffer is stored inline, `len` must be <= MAX_SIZE
    pub unsafe fn set_len(&mut self, len: usize) {
        match self {
            Repr::Static(s) => {
                *s = &s[..len];
            }
            Repr::Inline(inline) => inline.set_len(len),
            Repr::Heap(heap) => heap.set_len(len),
            Repr::ThinHeap(thin_heap) => thin_heap.set_len(len),
        }
    }
}

impl Clone for Repr {
    #[inline]
    fn clone(&self) -> Self {
        #[inline(never)]
        fn clone_heap(this: &Repr) -> Repr {
            Repr::new(this.as_str())
        }

        // There are only two cases we need to care about: If the string is allocated on the heap
        // or not. If it is, then the data must be cloned proberly, otherwise we can simply copy
        // the `Repr`.
        if self.is_heap_allocated() {
            clone_heap(self)
        } else {
            // SAFETY: We just checked that `self` can be copied because it is an inline string or
            // a reference to a `&'static str`.
            unsafe { core::ptr::read(self) }
        }
    }
}

impl Extend<char> for Repr {
    #[inline]
    fn extend<T: IntoIterator<Item = char>>(&mut self, iter: T) {
        let mut iterator = iter.into_iter().peekable();

        // if the iterator is empty, no work needs to be done!
        if iterator.peek().is_none() {
            return;
        }
        let (lower_bound, _) = iterator.size_hint();

        self.reserve(lower_bound);
        iterator.for_each(|c| self.push_str(c.encode_utf8(&mut [0; 4])));
    }
}

impl<'a> Extend<&'a char> for Repr {
    fn extend<T: IntoIterator<Item = &'a char>>(&mut self, iter: T) {
        self.extend(iter.into_iter().copied());
    }
}

impl<'a> Extend<&'a str> for Repr {
    fn extend<T: IntoIterator<Item = &'a str>>(&mut self, iter: T) {
        iter.into_iter().for_each(|s| self.push_str(s));
    }
}

impl Extend<Box<str>> for Repr {
    fn extend<T: IntoIterator<Item = Box<str>>>(&mut self, iter: T) {
        iter.into_iter().for_each(move |s| self.push_str(&s));
    }
}

impl<'a> Extend<Cow<'a, str>> for Repr {
    fn extend<T: IntoIterator<Item = Cow<'a, str>>>(&mut self, iter: T) {
        iter.into_iter().for_each(move |s| self.push_str(&s));
    }
}

impl Extend<String> for Repr {
    fn extend<T: IntoIterator<Item = String>>(&mut self, iter: T) {
        iter.into_iter().for_each(move |s| self.push_str(&s));
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::{
        String,
        ToString,
    };
    use alloc::vec::Vec;

    use quickcheck_macros::quickcheck;
    use test_case::test_case;

    use super::{
        Repr,
        MAX_SIZE,
    };

    const EIGHTEEN_MB: usize = 18 * 1024 * 1024;
    const EIGHTEEN_MB_STR: &str = unsafe { core::str::from_utf8_unchecked(&[42; EIGHTEEN_MB]) };

    #[test_case("hello world!"; "inline")]
    #[test_case("this is a long string that should be stored on the heap"; "heap")]
    fn test_create(s: &'static str) {
        let repr = Repr::new(s);
        assert_eq!(repr.as_str(), s);
        assert_eq!(repr.len(), s.len());

        // test StaticStr variant
        let repr = Repr::from_static_str(s);
        assert_eq!(repr.as_str(), s);
        assert_eq!(repr.len(), s.len());
    }

    #[quickcheck]
    #[cfg_attr(miri, ignore)]
    fn quickcheck_create(s: String) {
        let repr = Repr::new(&s);
        assert_eq!(repr.as_str(), s);
        assert_eq!(repr.len(), s.len());
    }

    #[test_case(0; "empty")]
    #[test_case(10; "short")]
    #[test_case(64; "long")]
    #[test_case(EIGHTEEN_MB; "huge")]
    fn test_with_capacity(cap: usize) {
        let r = Repr::with_capacity(cap);
        assert!(r.capacity() >= MAX_SIZE);
        assert_eq!(r.len(), 0);
    }

    #[test_case(""; "empty")]
    #[test_case("abc"; "short")]
    #[test_case("hello world! I am a longer string 🦀"; "long")]
    fn test_from_utf8_valid(s: &'static str) {
        let bytes = s.as_bytes();
        let r = Repr::from_utf8(bytes).expect("valid UTF-8");

        assert_eq!(r.as_str(), s);
        assert_eq!(r.len(), s.len());
    }

    #[quickcheck]
    #[cfg_attr(miri, ignore)]
    fn quickcheck_from_utf8(buf: Vec<u8>) {
        match (core::str::from_utf8(&buf), Repr::from_utf8(&buf)) {
            (Ok(s), Ok(r)) => {
                assert_eq!(r.as_str(), s);
                assert_eq!(r.len(), s.len());
            }
            (Err(e), Err(r)) => assert_eq!(e, r),
            _ => panic!("core::str and Repr differ on what is valid UTF-8!"),
        }
    }

    #[test_case(String::new(), true; "empty should inline")]
    #[test_case(String::new(), false; "empty not inline")]
    #[test_case(String::with_capacity(10), true ; "empty with small capacity inline")]
    #[test_case(String::with_capacity(10), false ; "empty with small capacity not inline")]
    #[test_case(String::with_capacity(128), true ; "empty with large capacity inline")]
    #[test_case(String::with_capacity(128), false ; "empty with large capacity not inline")]
    #[test_case(String::from("nyc 🗽"), true; "short should inline")]
    #[test_case(String::from("nyc 🗽"), false ; "short not inline")]
    #[test_case(String::from("this is a really long string, which is intended"), true; "long")]
    #[test_case(String::from("this is a really long string, which is intended"), false; "long not inline")]
    #[test_case(EIGHTEEN_MB_STR.to_string(), true ; "huge should inline")]
    #[test_case(EIGHTEEN_MB_STR.to_string(), false ; "huge not inline")]
    fn test_from_string(s: String, try_to_inline: bool) {
        // note: when cloning a String it truncates capacity, which is why we measure these values
        // before cloning the string
        let s_len = s.len();
        let s_cap = s.capacity();
        let s_str = s.clone();

        let r = Repr::from_string(s, try_to_inline);

        assert_eq!(r.len(), s_len);
        assert_eq!(r.as_str(), s_str.as_str());

        if s_cap == 0 || (try_to_inline && s_len <= MAX_SIZE) {
            // we should inline the string, if we were asked to, and the length of the string would
            // fit inline, meaning we would truncate capacity
            assert!(!r.is_heap_allocated());
        } else {
            assert!(r.is_heap_allocated());
        }
    }

    #[quickcheck]
    #[cfg_attr(miri, ignore)]
    fn quickcheck_from_string(s: String, try_to_inline: bool) {
        let r = Repr::from_string(s.clone(), try_to_inline);

        assert_eq!(r.len(), s.len());
        assert_eq!(r.as_str(), s.as_str());

        if s.capacity() == 0 {
            // we should always inline the string, if the length of the source string is 0
            assert!(!r.is_heap_allocated());
        } else if s.capacity() <= MAX_SIZE {
            // we should inline the string, if we were asked to
            assert_eq!(!r.is_heap_allocated(), try_to_inline);
        } else {
            assert!(r.is_heap_allocated());
        }
    }

    #[test_case(""; "empty")]
    #[test_case("nyc 🗽"; "short")]
    #[test_case("this is a really long string, which is intended"; "long")]
    fn test_into_string(control: &'static str) {
        let r = Repr::new(control);
        let s = r.into_string();

        assert_eq!(control.len(), s.len());
        assert_eq!(control, s.as_str());

        // test StaticStr variant
        let r = Repr::from_static_str(control);
        let s = r.into_string();

        assert_eq!(control.len(), s.len());
        assert_eq!(control, s.as_str());
    }

    #[quickcheck]
    #[cfg_attr(miri, ignore)]
    fn quickcheck_into_string(control: String) {
        let r = Repr::new(&control);
        let s = r.into_string();

        assert_eq!(control.len(), s.len());
        assert_eq!(control, s.as_str());
    }

    #[test_case("", "a", false; "empty")]
    #[test_case("", "🗽", false; "empty_emoji")]
    #[test_case("abc", "🗽🙂🦀🌈👏🐶", true; "inline_to_heap")]
    #[test_case("i am a long string that will be on the heap", "extra", true; "heap_to_heap")]
    fn test_push_str(control: &'static str, append: &'static str, is_heap: bool) {
        let mut r = Repr::new(control);
        let mut c = String::from(control);

        r.push_str(append);
        c.push_str(append);

        assert_eq!(r.as_str(), c.as_str());
        assert_eq!(r.len(), c.len());

        assert_eq!(r.is_heap_allocated(), is_heap);

        // test StaticStr variant
        let mut r = Repr::from_static_str(control);
        let mut c = String::from(control);

        r.push_str(append);
        c.push_str(append);

        assert_eq!(r.as_str(), c.as_str());
        assert_eq!(r.len(), c.len());

        assert_eq!(r.is_heap_allocated(), is_heap);
    }

    #[quickcheck]
    #[cfg_attr(miri, ignore)]
    fn quickcheck_push_str(control: String, append: String) {
        let mut r = Repr::new(&control);
        let mut c = control;

        r.push_str(&append);
        c.push_str(&append);

        assert_eq!(r.as_str(), c.as_str());
        assert_eq!(r.len(), c.len());
    }

    #[test_case(&[42; 0], &[42; EIGHTEEN_MB]; "empty_to_heap_capacity")]
    #[test_case(&[42; 8], &[42; EIGHTEEN_MB]; "inline_to_heap_capacity")]
    #[test_case(&[42; 128], &[42; EIGHTEEN_MB]; "heap_inline_to_heap_capacity")]
    #[test_case(&[42; EIGHTEEN_MB], &[42; 64]; "heap_capacity_to_heap_capacity")]
    fn test_push_str_from_buf(buf: &[u8], append: &[u8]) {
        // The goal of this test is to exercise the scenario when our capacity is stored on the heap

        let control = unsafe { core::str::from_utf8_unchecked(buf) };
        let append = unsafe { core::str::from_utf8_unchecked(append) };

        let mut r = Repr::new(control);
        let mut c = String::from(control);

        r.push_str(append);
        c.push_str(append);

        assert_eq!(r.as_str(), c.as_str());
        assert_eq!(r.len(), c.len());

        assert!(r.is_heap_allocated());
    }

    #[test_case("", 0, false; "empty_zero")]
    #[test_case("", 10, false; "empty_small")]
    #[test_case("", 64, true; "empty_large")]
    #[test_case("abc", 0, false; "short_zero")]
    #[test_case("abc", 8, false; "short_small")]
    #[test_case("abc", 64, true; "short_large")]
    #[test_case("I am a long string that will be on the heap", 0, true; "large_zero")]
    #[test_case("I am a long string that will be on the heap", 10, true; "large_small")]
    #[test_case("I am a long string that will be on the heap", EIGHTEEN_MB, true; "large_huge")]
    fn test_reserve(initial: &'static str, additional: usize, is_heap: bool) {
        let mut r = Repr::new(initial);
        r.reserve(additional);

        assert!(r.capacity() >= initial.len() + additional);
        assert_eq!(r.is_heap_allocated(), is_heap);

        // Test static_str variant
        let mut r = Repr::from_static_str(initial);
        r.reserve(additional);

        assert!(r.capacity() >= initial.len() + additional);
        assert_eq!(r.is_heap_allocated(), is_heap);
    }

    #[test]
    #[should_panic(expected = "Attempted to reserve more than 'usize' bytes")]
    fn test_reserve_overflow() {
        let mut r = Repr::new("abc");
        r.reserve(usize::MAX);
    }

    #[test_case(""; "empty")]
    #[test_case("abc"; "short")]
    #[test_case("i am a longer string that will be on the heap"; "long")]
    #[test_case(EIGHTEEN_MB_STR; "huge")]
    fn test_clone(initial: &'static str) {
        let r_a = Repr::new(initial);
        let r_b = r_a.clone();

        assert_eq!(r_a.as_str(), initial);
        assert_eq!(r_a.len(), initial.len());

        assert_eq!(r_a.as_str(), r_b.as_str());
        assert_eq!(r_a.len(), r_b.len());
        assert_eq!(r_a.capacity(), r_b.capacity());
        assert_eq!(r_a.is_heap_allocated(), r_b.is_heap_allocated());

        // test StaticStr variant
        let r_a = Repr::from_static_str(initial);
        let r_b = r_a.clone();

        assert_eq!(r_a.as_str(), initial);
        assert_eq!(r_a.len(), initial.len());

        assert_eq!(r_a.as_str(), r_b.as_str());
        assert_eq!(r_a.len(), r_b.len());
        assert_eq!(r_a.capacity(), r_b.capacity());
        assert_eq!(r_a.is_heap_allocated(), r_b.is_heap_allocated());
    }

    #[quickcheck]
    #[cfg_attr(miri, ignore)]
    fn quickcheck_clone(initial: String) {
        let r_a = Repr::new(&initial);
        let r_b = r_a.clone();

        assert_eq!(r_a.as_str(), initial);
        assert_eq!(r_a.len(), initial.len());

        assert_eq!(r_a.as_str(), r_b.as_str());
        assert_eq!(r_a.len(), r_b.len());
        assert_eq!(r_a.capacity(), r_b.capacity());
        assert_eq!(r_a.is_heap_allocated(), r_b.is_heap_allocated());
    }
}
