use deepviewrt_sys as ffi;
use std::{ffi::CStr, fmt, io};

#[derive(Debug, Clone)]
pub enum Error {
    NNError(&'static str),
    WrapperError(String),
    Null(),
    IoError(io::ErrorKind),
}

impl From<ffi::NNError> for Error {
    fn from(value: ffi::NNError) -> Self {
        let ret = unsafe { ffi::nn_strerror(value) };
        if ret.is_null() {
            return Error::Null();
        }
        let desc = unsafe { CStr::from_ptr(ret) };
        match desc.to_str() {
            Ok(estr) => return Error::NNError(estr),
            Err(_) => return Error::Null(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NNError(e) => return write!(f, "{}", e),
            Error::WrapperError(e) => return write!(f, "{}", e),
            Error::Null() => return write!(f, "null/unknown error message unavailable"),
            Error::IoError(kind) => {
                let e = std::io::Error::from(*kind);
                return write!(f, "{}", e);
            }
        }
    }
}

impl std::error::Error for Error {}
