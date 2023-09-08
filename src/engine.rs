use crate::error::Error;
use deepviewrt_sys as ffi;
use std::{
    ffi::{CStr, CString},
    path::Path,
};

pub struct Engine {
    ptr: *mut ffi::NNEngine,
}

impl Engine {
    pub fn new<P: AsRef<Path> + Into<Vec<u8>>>(path: P) -> Result<Self, Error> {
        let init_ret = unsafe { ffi::nn_engine_init(std::ptr::null_mut()) };
        if init_ret.is_null() {
            return Err(Error::WrapperError(
                "nn_engine_init memory allocated failed".to_string(),
            ));
        }
        let engine = Self { ptr: init_ret };

        let engine_cstring = CString::new(path);
        if let Err(e) = engine_cstring {
            return Err(Error::WrapperError(e.to_string()));
        }
        let load_ret = unsafe { ffi::nn_engine_load(init_ret, engine_cstring.unwrap().into_raw()) };
        if load_ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(load_ret));
        }
        return Ok(engine);
    }

    pub fn wrap(ptr: *mut ffi::NNEngine) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::Null());
        }
        return Ok(Engine { ptr });
    }

    pub fn name(&self) -> Option<&str> {
        let ret = unsafe { ffi::nn_engine_name(self.ptr) };
        if ret.is_null() {
            return None;
        }
        let name_cstr = unsafe { CStr::from_ptr(ret) };
        return Some(name_cstr.to_str().unwrap());
    }

    pub fn version(&self) -> Option<&str> {
        let ret = unsafe { ffi::nn_engine_version(self.ptr) };
        if ret.is_null() {
            return None;
        }
        let version_cstr = unsafe { CStr::from_ptr(ret) };
        return Some(version_cstr.to_str().unwrap());
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::nn_engine_release(self.ptr) };
    }
}
