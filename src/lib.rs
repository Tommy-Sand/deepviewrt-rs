use deepviewrt_sys as ffi;
pub mod context;
pub mod tensor;
pub mod engine;
pub mod error;
pub mod model;
use std::ffi::CStr;

pub enum QuantizationType {
	TypeNone = 0,
	TypeAffinePerTensor = 1,
	TypeAffinePerChannel = 2,
}

pub fn version() -> &'static str {
	let version = unsafe { ffi::nn_version() };
	let ret_cstr = unsafe { CStr::from_ptr(version) };
	return ret_cstr.to_str().unwrap();
}

pub fn init() {
	
}
