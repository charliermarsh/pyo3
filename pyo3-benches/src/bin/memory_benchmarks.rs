use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use num_bigint::BigUint;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{IntoPyDict, PyByteArray, PyModule, PySet};
use pyo3::IntoPyObjectExt;

const SET_LEN: usize = 100_000;
const DICT_LEN: usize = 50_000;
const PAYLOAD_BYTES: usize = 1 << 20;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static MEASURING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

impl CountingAllocator {
    fn record_allocation(size: usize) {
        if !MEASURING.load(Ordering::Relaxed) {
            return;
        }

        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
        let live_bytes = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
        PEAK_LIVE_BYTES.fetch_max(live_bytes, Ordering::Relaxed);
    }

    fn record_deallocation(size: usize) {
        if MEASURING.load(Ordering::Relaxed) {
            let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                Some(live.saturating_sub(size))
            });
        }
    }
}

// SAFETY: Allocation, deallocation, and reallocation are delegated unchanged to the
// system allocator. The counters never allocate and only observe successful operations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc`'s layout requirements.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            Self::record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc_zeroed`'s layout requirements.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            Self::record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        Self::record_deallocation(layout.size());
        // SAFETY: The caller provides the pointer and layout from the original allocation.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::realloc`'s pointer and layout requirements.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            Self::record_deallocation(layout.size());
            Self::record_allocation(new_size);
        }
        new_pointer
    }
}

fn measure<T>(
    tracemalloc: &Bound<'_, PyModule>,
    name: &str,
    operation: impl FnOnce() -> PyResult<T>,
) -> PyResult<()> {
    tracemalloc.call_method0("clear_traces")?;
    tracemalloc.call_method0("reset_peak")?;

    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    MEASURING.store(true, Ordering::Relaxed);
    let result = operation();
    MEASURING.store(false, Ordering::Relaxed);

    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let peak_live_bytes = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    let result = result?;
    let (_, python_peak_bytes) = tracemalloc
        .call_method0("get_traced_memory")?
        .extract::<(usize, usize)>()?;

    black_box(&result);
    println!("{name},{allocations},{allocated_bytes},{peak_live_bytes},{python_peak_bytes}");
    Ok(())
}

fn main() -> PyResult<()> {
    Python::attach(|py| {
        let python_set = PySet::new(py, 0..SET_LEN)?;
        let dict_items = (0..DICT_LEN)
            .map(|value| value.into_py_any(py))
            .collect::<PyResult<Vec<_>>>()?;
        let payload = vec![0xab; PAYLOAD_BYTES];
        let bytearray = PyByteArray::new(py, &payload);
        let large_integer = BigUint::from_bytes_le(&payload);

        let tracemalloc = PyModule::import(py, "tracemalloc")?;
        tracemalloc.call_method1("start", (1,))?;

        println!(
            "benchmark,rust_allocations,rust_allocated_bytes,rust_peak_live_bytes,python_peak_bytes"
        );

        measure(&tracemalloc, "extract_hashset", || {
            python_set.extract::<HashSet<u64>>()
        })?;
        measure(&tracemalloc, "extract_hashbrown_set", || {
            python_set.extract::<hashbrown::HashSet<u64>>()
        })?;
        measure(&tracemalloc, "construct_dict", || {
            dict_items
                .iter()
                .map(|value| (value, value))
                .into_py_dict(py)
        })?;
        measure(&tracemalloc, "bytearray_to_pybacked_bytes", || {
            Ok(PyBackedBytes::from(bytearray.clone()))
        })?;
        measure(&tracemalloc, "biguint_into_python", || {
            (&large_integer).into_pyobject(py)
        })?;

        tracemalloc.call_method0("stop")?;
        Ok(())
    })
}
