use deepviewrt_sys as ffi;
use std::{ffi::c_void, io, ops::Deref};
pub mod error;
use error::Error;

pub enum NNTensorType {
    RAW = 0,
    STR = 1,
    I8 = 2,
    U8 = 3,
    I16 = 4,
    U16 = 5,
    I32 = 6,
    U32 = 7,
    I64 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    F64 = 12,
}

pub struct NNTensor {
    ptr: *mut ffi::NNTensor,
}

unsafe impl Send for NNTensor {}

impl Deref for NNTensor {
    type Target = ffi::NNTensor;

    fn deref(&self) -> &Self::Target {
        return unsafe { &*(self.ptr) };
    }
}

impl NNTensor {
    pub fn new() -> Result<Self, Error> {
        let ptr = unsafe {
            ffi::nn_tensor_init(
                std::ptr::null::<c_void>() as *mut c_void,
                std::ptr::null::<ffi::nn_engine>() as *mut ffi::nn_engine,
            )
        };
        if ptr.is_null() {
            let err_kind = io::Error::last_os_error().kind();
            return Err(Error::IoError(err_kind));
        }

        return Ok(NNTensor { ptr });
    }

    pub fn alloc(&self, ttype: NNTensorType, n_dims: i32, shape: &[i32; 3]) -> Result<(), Error> {
        let ttype_c_uint = (ttype as u32) as std::os::raw::c_uint;
        let ret = unsafe { ffi::nn_tensor_alloc(self.ptr, ttype_c_uint, n_dims, shape.as_ptr()) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }

        return Ok(());
    }

    pub fn shape(&self) -> &[i32] {
        let temp = unsafe { ffi::nn_tensor_shape(self.ptr) };

        let ra = unsafe { std::slice::from_raw_parts(temp, 4) };
        return ra;
    }

    pub fn mapro_u8(&self, height: i32, width: i32, depth: i32) -> Result<&[u8], Error> {
        let ret = unsafe { ffi::nn_tensor_mapro(self.ptr) };
        if ret.is_null() {
            return Err(Error::WrapperError("nn_tensor_mapro failed".to_string()));
        }
        let ptr = ret as *const u8;
        let sret = unsafe { std::slice::from_raw_parts(ptr, (width * height * depth) as usize) };
        return Ok(sret);
    }

    pub fn unmap(&self) {
        unsafe { ffi::nn_tensor_unmap(self.ptr) };
    }

    pub fn to_mut_ptr(&self) -> *mut ffi::NNTensor {
        return self.ptr;
    }
}

impl Drop for NNTensor {
    fn drop(&mut self) {
        unsafe {
            ffi::nn_tensor_release(self.ptr);
        };
    }
}
