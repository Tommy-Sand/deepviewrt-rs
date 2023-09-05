use crate::{error::Error, tensor::NNTensor};
use deepviewrt_sys as ffi;

pub struct NNContext {
    ptr: *mut ffi::NNContext,
}

impl NNContext {
    pub unsafe fn from_ptr(ptr: *mut ffi::NNContext) -> Result<NNContext, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from("ptr is null")));
        }

        return Ok(NNContext { ptr });
    }

    pub fn tensor_index(&self, index: usize) -> Result<NNTensor, Error> {
        let ret = unsafe { ffi::nn_context_tensor_index(self.ptr, index) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("tensor is null")));
        }

        let tensor = unsafe { NNTensor::from_ptr(ret, false).unwrap() };
        return Ok(tensor);
    }
}
