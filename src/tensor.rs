use crate::{engine::Engine, error::Error};
use deepviewrt_sys as ffi;
use std::{ffi::c_void, io, ops::Deref};

#[derive(Debug)]
pub enum TensorType {
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

impl TryFrom<u32> for TensorType {
    type Error = ();

    fn try_from(value: u32) -> Result<TensorType, Self::Error> {
        match value {
            0 => return Ok(TensorType::RAW),
            1 => return Ok(TensorType::STR),
            2 => return Ok(TensorType::I8),
            3 => return Ok(TensorType::U8),
            4 => return Ok(TensorType::I16),
            5 => return Ok(TensorType::U16),
            6 => return Ok(TensorType::I32),
            7 => return Ok(TensorType::U32),
            8 => return Ok(TensorType::I64),
            9 => return Ok(TensorType::U64),
            10 => return Ok(TensorType::F16),
            11 => return Ok(TensorType::F32),
            12 => return Ok(TensorType::F64),
            _ => return Err(()),
        };
    }
}

pub struct NNTensor {
    owned: bool,
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

        return Ok(Self { owned: true, ptr });
    }

    pub fn alloc(&self, ttype: TensorType, n_dims: i32, shape: &[i32; 3]) -> Result<(), Error> {
        let ttype_c_uint = (ttype as u32) as std::os::raw::c_uint;
        let ret = unsafe { ffi::nn_tensor_alloc(self.ptr, ttype_c_uint, n_dims, shape.as_ptr()) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }

        return Ok(());
    }

    pub fn dequantize(&self, dest: &mut NNTensor) -> Result<(), Error> {
        let ret = unsafe { ffi::nn_tensor_dequantize(dest.to_mut_ptr(), self.ptr) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }

        return Ok(());
    }

    pub fn tensor_type(&self) -> TensorType {
        let ret = unsafe { ffi::nn_tensor_type(self.ptr) };
        return TensorType::try_from(ret).unwrap();
    }

    pub fn engine(&self) -> Option<Engine> {
        let ret = unsafe { ffi::nn_tensor_engine(self.ptr) };
        if ret.is_null() {
            return None;
        }
        return Some(Engine::wrap(ret).unwrap());
    }

    pub fn shape(&self) -> &[i32] {
        let ret = unsafe { ffi::nn_tensor_shape(self.ptr) };
        let ra = unsafe { std::slice::from_raw_parts(ret, 4) };
        return ra;
    }

    pub fn volume(&self) -> i32 {
        return unsafe { ffi::nn_tensor_volume(self.ptr) };
    }

    pub fn size(&self) -> i32 {
        return unsafe { ffi::nn_tensor_size(self.ptr) };
    }

    pub fn mapro_u8(&self) -> Result<&[u8], Error> {
        let ret = unsafe { ffi::nn_tensor_mapro(self.ptr) };
        if ret.is_null() {
            return Err(Error::WrapperError("nn_tensor_mapro failed".to_string()));
        }
        let ptr = ret as *const u8;
        let vol = self.volume();
        let sret = unsafe { std::slice::from_raw_parts(ptr, vol as usize) };
        return Ok(sret);
    }

    /*
    pub fn mapwo_u8(&self, height: i32, width: i32, depth: i32) -> Result<&mut [u8], Error> {

    }
    */

    pub fn unmap(&self) {
        unsafe { ffi::nn_tensor_unmap(self.ptr) };
    }

    pub unsafe fn from_ptr(ptr: *mut ffi::NNTensor, owned: bool) -> Result<NNTensor, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from("ptr is null")));
        }

        return Ok(NNTensor { owned, ptr });
    }

    pub fn to_mut_ptr(&self) -> *mut ffi::NNTensor {
        return self.ptr;
    }
}

impl Drop for NNTensor {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                ffi::nn_tensor_release(self.ptr);
            };
        }
    }
}
