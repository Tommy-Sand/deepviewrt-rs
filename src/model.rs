use crate::error::Error;
use deepviewrt_sys as ffi;
use std::ffi::{CStr, CString};

pub struct Model {
    ptr: *const ffi::NNModel,
}

impl Model {
    pub unsafe fn try_from_ptr(ptr: *const ffi::NNModel) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from(
                "try_from_ptr: pointer is null",
            )));
        }

        return Ok(Self { ptr });
    }

    pub fn name(&self) -> Result<&str, Error> {
        let ret = unsafe { ffi::nn_model_name(self.ptr) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("nn_model_name is null")));
        }
        let cstr = unsafe { CStr::from_ptr(ret) };
        match cstr.to_str() {
            Ok(s) => Ok(s),
            Err(e) => Err(Error::WrapperError(e.to_string())),
        }
    }

    /*
    pub fn serial(&self) -> Result<&str, Error> {

    }
    */

    pub fn label_count(&self) -> Result<i32, Error> {
        let ret = unsafe { ffi::nn_model_label_count(self.ptr) };
        if ret == 0 {
            return Err(Error::WrapperError(String::from(
                "No labels or model is invalid",
            )));
        }
        Ok(ret)
    }

    pub fn label(&self, index: i32) -> Result<&str, Error> {
        let ret = unsafe { ffi::nn_model_label(self.ptr, index) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("label was NULL")));
        }
        let cstr = unsafe { CStr::from_ptr(ret) };
        match cstr.to_str() {
            Ok(s) => Ok(s),
            Err(e) => Err(Error::WrapperError(e.to_string())),
        }
    }

    pub fn inputs(&self) -> Result<&[u32], Error> {
        let mut len: usize = 0;

        let ret = unsafe { ffi::nn_model_inputs(self.ptr, &mut len as *mut usize) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from(
                "could not get model inputs",
            )));
        }
        let outputs = unsafe { std::slice::from_raw_parts(ret, len) };
        return Ok(outputs);
    }

    pub fn outputs(&self) -> Result<&[u32], Error> {
        let mut n_outputs: usize = 0;
        let ret = unsafe { ffi::nn_model_outputs(self.ptr, &mut n_outputs as *mut usize) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from(
                "could not get model outputs",
            )));
        }
        let outputs = unsafe { std::slice::from_raw_parts(ret, n_outputs) };
        return Ok(outputs);
    }

    pub fn layer_count(&self) -> usize {
        return unsafe { ffi::nn_model_layer_count(self.ptr) };
    }

    pub fn layer_name(&self, index: usize) -> Result<&str, Error> {
        let ret = unsafe { ffi::nn_model_layer_name(self.ptr, index) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from(
                "nn_model_layer_name returned null",
            )));
        }
        let cstr = unsafe { CStr::from_ptr(ret) };
        match cstr.to_str() {
            Ok(s) => Ok(s),
            Err(e) => Err(Error::WrapperError(e.to_string())),
        }
    }

    pub fn layer_lookup(&self, name: &str) -> Result<i32, Error> {
        let name = match CString::new(name) {
            Ok(s) => s,
            Err(e) => return Err(Error::WrapperError(e.to_string())),
        };

        let ret = unsafe { ffi::nn_model_layer_lookup(self.ptr, name.into_raw()) };
        if ret == -1 {
            return Err(Error::WrapperError(String::from(
                "Could not get index of layer",
            )));
        }
        return Ok(ret);
    }

    pub fn layer_type(&self, index: usize) -> Result<&str, Error> {
        let ret = unsafe { ffi::nn_model_layer_type(self.ptr, index) };
		if ret.is_null() {
			return Err(Error::WrapperError(String::from("index out of range")));
		}

		let cstr = unsafe { CStr::from_ptr(ret) };
		match cstr.to_str() {
			Ok(s) => Ok(s),
			Err(e) => Err(Error::WrapperError(e.to_string())),
		}
    }

	/*
    pub fn layer_type_id(&self, index: usize) -> Result<i16, Error> {
		let ret = unsafe { ffi::nn_model_layer_type_id(self.ptr, index) };
		if ret == 0 {
			return Err(Error::WrapperError(String::from("index out of range")));
		}
    }
	*/

    pub fn layer_datatype(&self, index: usize) -> Result<&str, Error> {
		let ret = unsafe { ffi::nn_model_layer_datatype(self.ptr, index) };
		if ret.is_null() {
			return Err(Error::WrapperError(String::from("index out of range")));
		}

		let cstr = unsafe { CStr::from_ptr(ret) };
		match cstr.to_str() {
			Ok(s) => Ok(s),
			Err(e) => Err(Error::WrapperError(e.to_string())),
		}
    }

	pub fn layer_datatype_id() { }

	pub fn layer_zeros() { }

	pub fn layer_scales() { }

	pub fn layer_axis() { }

	pub fn layer_shape() { }

	pub fn layer_inputs() { }

	pub fn layer_parameter() { }

	pub fn layer_parameter_shape() { }
}
