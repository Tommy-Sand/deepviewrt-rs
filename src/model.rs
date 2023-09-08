use crate::error::Error;
use deepviewrt_sys as ffi;

pub struct Model {
    ptr: *const ffi::NNModel,
}

impl Model {
    pub fn try_from_ptr(ptr: *const ffi::NNModel) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from(
                "try_from_ptr: pointer is null",
            )));
        }

        return Ok(Self { ptr });
    }

    pub fn inputs(&self) -> Result<&[u32], Error> {
        let mut len: usize = 0;

        let ret = unsafe { ffi::nn_model_inputs(self.ptr, &mut len as *mut usize) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from(
                "could not get model inputs",
            )));
        }

        return Ok(unsafe { std::slice::from_raw_parts(ret, len) });
    }
}
