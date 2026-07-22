use crate::platform::prelude::*;
use crate::types::any::PyAnyMethods;
use crate::Bound;
use crate::{exceptions::PyTypeError, FromPyObject, PyAny, PyErr, PyResult, Python};

/// Keeps speculative enum extraction errors unwrapped until every variant has failed.
pub enum DeferredFromPyObjectError {
    Plain(PyErr),
    StructField {
        error: PyErr,
        struct_name: &'static str,
        field_name: &'static str,
    },
    TupleField {
        error: PyErr,
        struct_name: &'static str,
        index: usize,
    },
}

impl From<PyErr> for DeferredFromPyObjectError {
    fn from(error: PyErr) -> Self {
        Self::Plain(error)
    }
}

impl DeferredFromPyObjectError {
    fn normalize_field_error(py: Python<'_>, error: PyErr) -> PyErr {
        // Field-error wrapping previously normalized its cause immediately, so retain
        // that observable timing while deferring construction of the wrapper itself.
        let _ = error.value(py);
        error
    }

    fn clone_as_pyerr(&self, py: Python<'_>) -> PyErr {
        match self {
            Self::Plain(error) => error.clone_ref(py),
            Self::StructField {
                error,
                struct_name,
                field_name,
            } => failed_to_extract_struct_field(py, error.clone_ref(py), struct_name, field_name),
            Self::TupleField {
                error,
                struct_name,
                index,
            } => failed_to_extract_tuple_struct_field(py, error.clone_ref(py), struct_name, *index),
        }
    }
}

#[cold]
pub fn failed_to_extract_enum(
    py: Python<'_>,
    type_name: &str,
    variant_names: &[&str],
    error_names: &[&str],
    errors: &[DeferredFromPyObjectError],
) -> PyErr {
    // TODO maybe use ExceptionGroup on Python 3.11+ ?
    let mut err_msg = format!(
        "failed to extract enum {} ('{}')",
        type_name,
        error_names.join(" | ")
    );
    for ((variant_name, error_name), error) in variant_names.iter().zip(error_names).zip(errors) {
        use core::fmt::Write;
        write!(
            &mut err_msg,
            "\n- variant {variant_name} ({error_name}): {error_msg}",
            variant_name = variant_name,
            error_name = error_name,
            error_msg = extract_traceback(py, error.clone_as_pyerr(py)),
        )
        .unwrap();
    }
    PyTypeError::new_err(err_msg)
}

/// Flattens a chain of errors into a single string.
fn extract_traceback(py: Python<'_>, mut error: PyErr) -> String {
    use core::fmt::Write;

    let mut error_msg = error.to_string();
    while let Some(cause) = error.cause(py) {
        write!(&mut error_msg, ", caused by {cause}").unwrap();
        error = cause
    }
    error_msg
}

pub fn extract_struct_field<'a, 'py, T>(
    obj: &'a Bound<'py, PyAny>,
    struct_name: &str,
    field_name: &str,
) -> PyResult<T>
where
    T: FromPyObject<'a, 'py>,
{
    match obj.extract() {
        Ok(value) => Ok(value),
        Err(err) => Err(failed_to_extract_struct_field(
            obj.py(),
            err.into(),
            struct_name,
            field_name,
        )),
    }
}

pub fn extract_struct_field_deferred<'a, 'py, T>(
    obj: &'a Bound<'py, PyAny>,
    struct_name: &'static str,
    field_name: &'static str,
) -> Result<T, DeferredFromPyObjectError>
where
    T: FromPyObject<'a, 'py>,
{
    obj.extract::<T>()
        .map_err(|error| DeferredFromPyObjectError::StructField {
            error: DeferredFromPyObjectError::normalize_field_error(obj.py(), error.into()),
            struct_name,
            field_name,
        })
}

pub fn extract_struct_field_with<'a, 'py, T>(
    extractor: fn(&'a Bound<'py, PyAny>) -> PyResult<T>,
    obj: &'a Bound<'py, PyAny>,
    struct_name: &str,
    field_name: &str,
) -> PyResult<T> {
    match extractor(obj) {
        Ok(value) => Ok(value),
        Err(err) => Err(failed_to_extract_struct_field(
            obj.py(),
            err,
            struct_name,
            field_name,
        )),
    }
}

pub fn extract_struct_field_with_deferred<'a, 'py, T>(
    extractor: fn(&'a Bound<'py, PyAny>) -> PyResult<T>,
    obj: &'a Bound<'py, PyAny>,
    struct_name: &'static str,
    field_name: &'static str,
) -> Result<T, DeferredFromPyObjectError> {
    extractor(obj).map_err(|error| DeferredFromPyObjectError::StructField {
        error: DeferredFromPyObjectError::normalize_field_error(obj.py(), error),
        struct_name,
        field_name,
    })
}

#[cold]
fn failed_to_extract_struct_field(
    py: Python<'_>,
    inner_err: PyErr,
    struct_name: &str,
    field_name: &str,
) -> PyErr {
    let new_err = PyTypeError::new_err(format!(
        "failed to extract field {struct_name}.{field_name}"
    ));
    new_err.set_cause(py, ::core::option::Option::Some(inner_err));
    new_err
}

pub fn extract_tuple_struct_field<'a, 'py, T>(
    obj: &'a Bound<'py, PyAny>,
    struct_name: &str,
    index: usize,
) -> PyResult<T>
where
    T: FromPyObject<'a, 'py>,
{
    match obj.extract() {
        Ok(value) => Ok(value),
        Err(err) => Err(failed_to_extract_tuple_struct_field(
            obj.py(),
            err.into(),
            struct_name,
            index,
        )),
    }
}

pub fn extract_tuple_struct_field_deferred<'a, 'py, T>(
    obj: &'a Bound<'py, PyAny>,
    struct_name: &'static str,
    index: usize,
) -> Result<T, DeferredFromPyObjectError>
where
    T: FromPyObject<'a, 'py>,
{
    obj.extract::<T>()
        .map_err(|error| DeferredFromPyObjectError::TupleField {
            error: DeferredFromPyObjectError::normalize_field_error(obj.py(), error.into()),
            struct_name,
            index,
        })
}

pub fn extract_tuple_struct_field_with<'a, 'py, T>(
    extractor: fn(&'a Bound<'py, PyAny>) -> PyResult<T>,
    obj: &'a Bound<'py, PyAny>,
    struct_name: &str,
    index: usize,
) -> PyResult<T> {
    match extractor(obj) {
        Ok(value) => Ok(value),
        Err(err) => Err(failed_to_extract_tuple_struct_field(
            obj.py(),
            err,
            struct_name,
            index,
        )),
    }
}

pub fn extract_tuple_struct_field_with_deferred<'a, 'py, T>(
    extractor: fn(&'a Bound<'py, PyAny>) -> PyResult<T>,
    obj: &'a Bound<'py, PyAny>,
    struct_name: &'static str,
    index: usize,
) -> Result<T, DeferredFromPyObjectError> {
    extractor(obj).map_err(|error| DeferredFromPyObjectError::TupleField {
        error: DeferredFromPyObjectError::normalize_field_error(obj.py(), error),
        struct_name,
        index,
    })
}

#[cold]
fn failed_to_extract_tuple_struct_field(
    py: Python<'_>,
    inner_err: PyErr,
    struct_name: &str,
    index: usize,
) -> PyErr {
    let new_err = PyTypeError::new_err(format!("failed to extract field {struct_name}.{index}"));
    new_err.set_cause(py, ::core::option::Option::Some(inner_err));
    new_err
}
