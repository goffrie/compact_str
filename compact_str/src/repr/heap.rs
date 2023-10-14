use core::mem::MaybeUninit;
use core::num::NonZeroUsize;
use core::{cmp, mem, ptr, slice};

use alloc::string::String;
use alloc::vec::Vec;

use super::capacity::Capacity;
use super::{Repr, MAX_SIZE};

/// The minimum size we'll allocate on the heap is one usize larger than our max inline size
const MIN_HEAP_SIZE: usize = MAX_SIZE + mem::size_of::<usize>();

const UNKNOWN: usize = 0;
pub type StrBuffer = [u8; UNKNOWN];

/// [`HeapBuffer`] grows at an amortized rates of 1.5x
///
/// Note: this is different than [`std::string::String`], which grows at a rate of 2x. It's debated
/// which is better, for now we'll stick with a rate of 1.5x
#[inline(always)]
pub fn amortized_growth(cur_len: usize, additional: usize) -> usize {
    let required = cur_len.saturating_add(additional);
    let amortized = cur_len.saturating_mul(3) / 2;
    amortized.max(required)
}

pub struct Cap([u8; mem::size_of::<usize>() - 1]);

pub struct TooLong;

impl Cap {
    #[inline]
    fn new(l: usize) -> Result<Self, TooLong> {
        if l < (1 << (mem::size_of::<usize>() - 1)) {
            Ok(Cap(l.to_le_bytes()[..mem::size_of::<usize>() - 1]
                .try_into()
                .unwrap()))
        } else {
            Err(TooLong)
        }
    }
}

impl From<Cap> for usize {
    #[inline]
    fn from(value: Cap) -> Self {
        let mut buf = [0; mem::size_of::<usize>()];
        buf[..mem::size_of::<usize>() - 1].copy_from_slice(&value.0);
        usize::from_le_bytes(buf)
    }
}

#[repr(C, packed)]
pub struct HeapBuffer {
    // Invariant:
    // `ptr` points to a heap-allocated buffer of length `cap` and alignment 1,
    // and its first `len` bytes are initialized with valid UTF-8;
    // i.e. these fields correspond to those of a `String`
    ptr: ptr::NonNull<u8>,
    len: usize,
    cap: Cap,
}
#[repr(C, packed)]
pub struct ThinHeapBuffer {
    ptr: ptr::NonNull<u8>,
    len: usize,
}

impl HeapBuffer {
    /// Create a [`HeapBuffer`] with the provided text. Returns `Err` if the text is too long.
    #[inline]
    pub fn new(text: &str) -> Result<Self, TooLong> {
        let capacity = text.len().max(MIN_HEAP_SIZE);
        let cap = Cap::new(capacity)?;
        let mut allocated = Vec::with_capacity(capacity);
        allocated.extend_from_slice(text.as_bytes()); // this cannot reallocate because `capacity` >= `text.len()`
        let ptr = ptr::NonNull::new(allocated.as_mut_ptr()).unwrap();
        let len = allocated.len();
        mem::forget(allocated);
        // SAFETY: `ptr` comes from an Vec allocated with capacity `` == `cap`,
        // and its content comes from a `&str`
        Ok(HeapBuffer { ptr, len, cap })
    }

    /// Create an empty [`HeapBuffer`] with a specific capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Result<Self, TooLong> {
        let capacity = capacity.max(MIN_HEAP_SIZE);
        let cap = Cap::new(capacity)?;
        let mut allocated = Vec::with_capacity(capacity);
        let ptr = ptr::NonNull::new(allocated.as_mut_ptr()).unwrap();
        mem::forget(allocated);
        // SAFETY: `ptr` comes from an Vec allocated with capacity `capacity` == `cap`
        Ok(HeapBuffer { ptr, len: 0, cap })
    }

    /// Create a [`HeapBuffer`] with `text` that has _at least_ `additional` bytes of capacity
    ///
    /// To prevent frequent re-allocations, this method will create a [`HeapBuffer`] with a capacity
    /// of `text.len() + additional` or `text.len() * 1.5`, whichever is greater
    #[inline]
    pub fn with_additional(text: &str, additional: usize) -> Result<Self, TooLong> {
        let new_capacity = amortized_growth(text.len(), additional).max(MIN_HEAP_SIZE);
        let new_cap = Cap::new(new_capacity)?;
        let mut allocated = Vec::with_capacity(new_capacity);
        allocated.extend_from_slice(text.as_bytes()); // this cannot reallocate because `capacity` >= `text.len()`
        let ptr = ptr::NonNull::new(allocated.as_mut_ptr()).unwrap();
        let len = allocated.len();
        mem::forget(allocated);
        // SAFETY: `ptr` comes from an Vec allocated with capacity `new_capacity` == `new_cap`
        // and its content comes from a `&str`
        Ok(HeapBuffer {
            ptr,
            len,
            cap: new_cap,
        })
    }

    /// Create a [`HeapBuffer`] with the provided `String`, reusing its allocation. Returns `Err` if its capacity
    /// is too large for this representation.
    #[inline]
    pub fn from_string(mut text: String) -> Result<Self, String> {
        let cap = match Cap::new(text.capacity()) {
            Ok(c) => c,
            Err(TooLong) => return Err(text),
        };
        let len = text.len();
        let ptr = ptr::NonNull::new(text.as_mut_ptr()).unwrap();
        mem::forget(text);
        // SAFETY: `ptr` comes from an String allocated with capacity `cap`,
        Ok(HeapBuffer { ptr, len, cap })
    }

    /// Return the capacity of the [`HeapBuffer`]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap.into()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Grow the [`HeapBuffer`].
    /// Returns an error if the capacity is too long for this representation.
    pub fn realloc(&mut self, new_capacity: usize) -> Result<usize, TooLong> {
        // Always allocate at least MIN_HEAP_SIZE
        let new_capacity = new_capacity.max(MIN_HEAP_SIZE);

        let new_cap = Cap::new(new_capacity)?;

        // We can't reallocate to a size less than our length, or else we'd clip the string
        assert!(new_capacity >= self.len);

        let mut vec = unsafe {
            // SAFETY: xxx
            mem::ManuallyDrop::new(Vec::from_raw_parts(
                self.ptr.as_ptr(),
                self.len,
                self.cap.into(),
            ))
        };

        if new_capacity > vec.capacity() {
            vec.reserve_exact(new_capacity - vec.capacity());
        } else {
            vec.shrink_to(new_capacity);
        }
        debug_assert_eq!(vec.capacity(), new_capacity);

        self.ptr = ptr::NonNull::new(vec.as_mut_ptr()).unwrap();
        self.cap = new_cap;
        Ok(new_capacity)
    }

    /// Set's the length of the content for this [`HeapBuffer`]
    ///
    /// # SAFETY:
    /// * The caller must guarantee that `self.capacity()` is at least `len`
    /// * The caller must guarantee that `len` bytes in the buffer are valid UTF-8
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    pub(crate) fn into_string(self) -> String {
        unsafe {
            // SAFETY: see this type's invariant
            let s = String::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap.into());
            mem::forget(self);
            s
        }
    }

    pub(crate) unsafe fn as_mut_buf(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: this type's invariant guarantees that self.ptr is allocated for at least self.len
        // the caller must guarantee that they only write valid UTF-8
        slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.cap.into())
    }
}

impl Clone for HeapBuffer {
    fn clone(&self) -> Self {
        // Create a new HeapBuffer with the same capacity as the original
        let mut new = Self::with_capacity(self.capacity());

        // SAFETY:
        // * `src` and `dst` don't overlap because we just created `dst`
        // * `src` and `dst` are both valid for `self.len` bytes because self.len < capacity
        unsafe {
            new.ptr
                .as_ptr()
                .copy_from_nonoverlapping(self.ptr.as_ptr(), self.len)
        };
        // SAFETY:
        // * We copied the text from self, which is valid UTF-8
        unsafe { new.set_len(self.len) };

        new
    }
}

impl Drop for HeapBuffer {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: see this type's invariant
            drop(Vec::from_raw_parts(
                self.ptr.as_ptr(),
                self.len,
                self.cap.into(),
            ))
        }
    }
}

/// Allocates a buffer on the heap that we can use to store a string, optionally stores the capacity
/// of said buffer on the heap.
///
/// Returns a [`Capacity`] that either indicates the capacity is stored on the heap, or is stored
/// in the `Capacity` itself.
#[inline]
pub fn allocate_ptr(capacity: usize) -> (Capacity, ptr::NonNull<u8>) {
    // We allocate at least MIN_HEAP_SIZE bytes because we need to allocate at least one byte
    let capacity = capacity.max(MIN_HEAP_SIZE);
    let cap = Capacity::new(capacity);

    // HeapBuffer doesn't support 0 sized allocations, we should always allocate at least
    // MIN_HEAP_SIZE bytes
    debug_assert!(capacity > 0);

    #[cold]
    fn allocate_with_capacity_on_heap(capacity: usize) -> ptr::NonNull<u8> {
        // write our capacity onto the heap
        let ptr = heap_capacity::alloc(capacity);
        // SAFETY:
        // * `src` and `dst` don't overlap and are both valid for `usize` bytes
        unsafe {
            ptr::copy_nonoverlapping(
                capacity.to_ne_bytes().as_ptr(),
                ptr.as_ptr(),
                mem::size_of::<usize>(),
            )
        };
        let raw_ptr = ptr.as_ptr().wrapping_add(core::mem::size_of::<usize>());
        // SAFETY: We know `raw_ptr` is non-null because we just created it
        unsafe { ptr::NonNull::new_unchecked(raw_ptr) }
    }

    let ptr = if cap.is_heap() {
        allocate_with_capacity_on_heap(capacity)
    } else {
        unsafe { inline_capacity::alloc(capacity) }
    };

    (cap, ptr)
}

/// Deallocates a buffer on the heap, handling when the capacity is also stored on the heap
#[inline]
pub fn deallocate_ptr(ptr: ptr::NonNull<u8>, cap: Capacity) {
    #[cold]
    fn deallocate_with_capacity_on_heap(ptr: ptr::NonNull<u8>) {
        // re-adjust the pointer to include the capacity that's on the heap
        let adj_ptr = ptr.as_ptr().wrapping_sub(mem::size_of::<usize>());
        // read the capacity from the heap so we know how much to deallocate
        let mut buf = [0u8; mem::size_of::<usize>()];
        // SAFETY: `src` and `dst` don't overlap, and are valid for usize number of bytes
        unsafe {
            ptr::copy_nonoverlapping(adj_ptr, buf.as_mut_ptr(), mem::size_of::<usize>());
        }
        let capacity = usize::from_ne_bytes(buf);
        // SAFETY: We know the pointer is not null since we got it as a NonNull
        let ptr = unsafe { ptr::NonNull::new_unchecked(adj_ptr) };
        // SAFETY: We checked above that our capacity is on the heap, and we readjusted the
        // pointer to reference the capacity
        unsafe { heap_capacity::dealloc(ptr, capacity) }
    }

    if cap.is_heap() {
        deallocate_with_capacity_on_heap(ptr);
    } else {
        // SAFETY: Our capacity is always inline on 64-bit archs
        unsafe { inline_capacity::dealloc(ptr, cap.as_usize()) }
    }
}

mod heap_capacity {
    use core::{alloc, ptr};

    use super::StrBuffer;

    #[inline]
    pub fn alloc(capacity: usize) -> ptr::NonNull<u8> {
        let layout = layout(capacity);
        debug_assert!(layout.size() > 0);

        // SAFETY: `alloc(...)` has undefined behavior if the layout is zero-sized. We know the
        // layout can't be zero-sized though because we're always at least allocating one `usize`
        let raw_ptr = unsafe { ::alloc::alloc::alloc(layout) };

        // Check to make sure our pointer is non-null, some allocators return null pointers instead
        // of panicking
        match ptr::NonNull::new(raw_ptr) {
            Some(ptr) => ptr,
            None => ::alloc::alloc::handle_alloc_error(layout),
        }
    }

    /// Deallocates a pointer which references a `HeapBuffer` whose capacity is on the heap
    ///
    /// # Saftey
    /// * `ptr` must point to the start of a `HeapBuffer` whose capacity is on the heap. i.e. we
    ///   must have `ptr -> [cap<usize> ; string<bytes>]`
    pub unsafe fn dealloc(ptr: ptr::NonNull<u8>, capacity: usize) {
        let layout = layout(capacity);
        ::alloc::alloc::dealloc(ptr.as_ptr(), layout);
    }

    #[repr(C)]
    struct HeapBufferInnerHeapCapacity {
        capacity: usize,
        buffer: StrBuffer,
    }

    #[inline(always)]
    pub fn layout(capacity: usize) -> alloc::Layout {
        let buffer_layout = alloc::Layout::array::<u8>(capacity).expect("valid capacity");
        alloc::Layout::new::<HeapBufferInnerHeapCapacity>()
            .extend(buffer_layout)
            .expect("valid layout")
            .0
            .pad_to_align()
    }
}

mod inline_capacity {
    use core::{alloc, ptr};

    use super::StrBuffer;

    /// # SAFETY:
    /// * `capacity` must be > 0
    #[inline]
    pub unsafe fn alloc(capacity: usize) -> ptr::NonNull<u8> {
        let layout = layout(capacity);
        debug_assert!(layout.size() > 0);

        // SAFETY: `alloc(...)` has undefined behavior if the layout is zero-sized. We specify that
        // `capacity` must be > 0 as a constraint to uphold the safety of this method. If capacity
        // is greater than 0, then our layout will be non-zero-sized.
        let raw_ptr = ::alloc::alloc::alloc(layout);

        // Check to make sure our pointer is non-null, some allocators return null pointers instead
        // of panicking
        match ptr::NonNull::new(raw_ptr) {
            Some(ptr) => ptr,
            None => ::alloc::alloc::handle_alloc_error(layout),
        }
    }

    /// Deallocates a pointer which references a `HeapBuffer` whose capacity is stored inline
    ///
    /// # Saftey
    /// * `ptr` must point to the start of a `HeapBuffer` whose capacity is on the inline
    pub unsafe fn dealloc(ptr: ptr::NonNull<u8>, capacity: usize) {
        let layout = layout(capacity);
        ::alloc::alloc::dealloc(ptr.as_ptr(), layout);
    }

    #[repr(C)]
    struct HeapBufferInnerInlineCapacity {
        buffer: StrBuffer,
    }

    #[inline(always)]
    pub fn layout(capacity: usize) -> alloc::Layout {
        let buffer_layout = alloc::Layout::array::<u8>(capacity).expect("valid capacity");
        alloc::Layout::new::<HeapBufferInnerInlineCapacity>()
            .extend(buffer_layout)
            .expect("valid layout")
            .0
            .pad_to_align()
    }
}

#[cfg(test)]
mod test {
    use test_case::test_case;

    use super::{HeapBuffer, MIN_HEAP_SIZE};

    const EIGHTEEN_MB: usize = 18 * 1024 * 1024;

    #[test]
    fn test_min_capacity() {
        let h = HeapBuffer::new("short");
        assert_eq!(h.capacity(), MIN_HEAP_SIZE);
    }

    #[test_case(&[42; 8]; "short")]
    #[test_case(&[42; 50]; "long")]
    #[test_case(&[42; EIGHTEEN_MB]; "huge")]
    fn test_capacity(buf: &[u8]) {
        // we know the buffer is valid UTF-8
        let s = unsafe { core::str::from_utf8_unchecked(buf) };
        let h = HeapBuffer::new(s);

        assert_eq!(h.capacity(), core::cmp::max(s.len(), MIN_HEAP_SIZE));
    }

    #[test_case(&[42; 0], 0, Err(MIN_HEAP_SIZE); "empty_empty")]
    #[test_case(&[42; 64], 0, Err(64); "short_empty")]
    #[test_case(&[42; 64], 32, Err(64); "short_to_shorter")]
    #[test_case(&[42; 64], 128, Ok(128); "short_to_longer")]
    #[test_case(&[42; EIGHTEEN_MB], EIGHTEEN_MB + 128, Ok(EIGHTEEN_MB + 128); "heap_to_heap")]
    fn test_realloc(buf: &[u8], realloc: usize, result: Result<usize, usize>) {
        // we know the buffer is valid UTF-8
        let s = unsafe { core::str::from_utf8_unchecked(buf) };
        let mut h = HeapBuffer::new(s);

        // reallocate, asserting our result
        let expected_cap = match result {
            Ok(c) | Err(c) => c,
        };
        let expected_res = result.map_err(|_| ());
        assert_eq!(h.realloc(realloc), expected_res);
        assert_eq!(h.capacity(), expected_cap);
    }

    #[test]
    fn test_realloc_inline_to_heap() {
        // we know the buffer is valid UTF-8
        let s = unsafe { core::str::from_utf8_unchecked(&[42; 128]) };
        let mut h = HeapBuffer::new(s);

        cfg_if::cfg_if! {
            if #[cfg(target_pointer_width = "64")] {
                let expected_result = Ok(EIGHTEEN_MB);
                let expected_capacity = EIGHTEEN_MB;
            } else if #[cfg(target_pointer_width = "32")] {
                // on 32-bit architectures we'd need to change the layout from capacity being inline
                // to the capacity being on the heap, which isn't possible
                let expected_result = Err(());
                let expected_capacity = 128;
            } else {
                compile_error!("Unsupported pointer width!");
            }
        }
        assert_eq!(h.realloc(EIGHTEEN_MB), expected_result);
        assert_eq!(h.capacity(), expected_capacity);
    }

    #[test_case(&[42; 64], 128, 100, Ok(100); "sanity")]
    fn test_realloc_shrink(
        buf: &[u8],
        realloc_one: usize,
        realloc_two: usize,
        exp_result: Result<usize, usize>,
    ) {
        // we know the buffer is valid UTF-8
        let s = unsafe { core::str::from_utf8_unchecked(buf) };
        let mut h = HeapBuffer::new(s);

        assert!(
            realloc_one > realloc_two,
            "we have to grow before we can shrink"
        );

        // grow our allocation
        assert_eq!(h.realloc(realloc_one), Ok(realloc_one));

        // shrink our allocation, asserting our result
        let expected_cap = match exp_result {
            Ok(c) | Err(c) => c,
        };
        let expected_res = exp_result.map_err(|_| ());
        assert_eq!(h.realloc(realloc_two), expected_res);
        assert_eq!(h.capacity(), expected_cap);
    }

    #[test]
    fn test_realloc_shrink_heap_to_inline() {
        // TODO: test this case
        assert_eq!(1, 1);
    }

    #[test_case(&[42; 0]; "empty")]
    #[test_case(&[42; 3]; "short")]
    #[test_case(&[42; 64]; "long")]
    #[test_case(&[42; EIGHTEEN_MB]; "huge")]
    fn test_clone(buf: &[u8]) {
        let s = unsafe { core::str::from_utf8_unchecked(buf) };
        let h_a = HeapBuffer::new(s);
        let h_b = h_a.clone();

        assert_eq!(h_a.capacity(), h_b.capacity());
    }
}
