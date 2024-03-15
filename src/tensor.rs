use crate::{engine::Engine, error::Error};
use deepviewrt_sys as ffi;
use std::{
    cell::Cell,
    ffi::{c_void, CStr},
    io,
    ops::Deref,
};

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

pub struct Tensor {
    owned: bool,
    ptr: *mut ffi::NNTensor,
    engine: Cell<Option<Engine>>,
    scales: Option<Vec<f32>>,
}

#[repr(u8)]
pub enum MappedData<'a> {
    RAW(&'a [u8]) = 0,
    STR(&'a str) = 1,
    I8(&'a [i8]) = 2,
    U8(&'a [u8]) = 3,
    I16(&'a [i16]) = 4,
    U16(&'a [u16]) = 5,
    I32(&'a [i32]) = 6,
    U32(&'a [u32]) = 7,
    I64(&'a [i64]) = 8,
    U64(&'a [u64]) = 9,
    F16(&'a [u8]) = 10,
    F32(&'a [f32]) = 11,
    F64(&'a [f64]) = 12,
}

pub struct TensorData<'a> {
    tensor: &'a Tensor,
    data: MappedData<'a>,
}

impl<'a> Deref for TensorData<'a> {
    type Target = MappedData<'a>;

    fn deref(&self) -> &Self::Target {
        return &self.data;
    }
}

impl<'a> Drop for TensorData<'a> {
    fn drop(&mut self) {
        unsafe { self.tensor.unmap() };
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Deref for Tensor {
    type Target = ffi::NNTensor;

    fn deref(&self) -> &Self::Target {
        return unsafe { &*(self.ptr) };
    }
}

impl Tensor {
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

        return Ok(Self {
            owned: true,
            engine: Cell::new(None),
            ptr,
            scales: None,
        });
    }

    pub fn alloc(&self, ttype: TensorType, n_dims: i32, shape: &[i32; 3]) -> Result<(), Error> {
        let ttype_c_uint = (ttype as u32) as std::os::raw::c_uint;
        let ret = unsafe { ffi::nn_tensor_alloc(self.ptr, ttype_c_uint, n_dims, shape.as_ptr()) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }

        return Ok(());
    }

    pub fn dequantize(&self, dest: &mut Self) -> Result<(), Error> {
        let ret = unsafe { ffi::nn_tensor_dequantize(dest.to_mut_ptr(), self.ptr) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }

        return Ok(());
    }

    pub fn set_tensor_type(&self, tensor_type: TensorType) -> Result<(), Error> {
        let tensor_type_ = TensorType::try_from(tensor_type as u32).unwrap();
        let ret = unsafe { ffi::nn_tensor_set_type(self.ptr, tensor_type_ as ffi::NNTensorType) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }
        return Ok(());
    }

    pub fn tensor_type(&self) -> TensorType {
        let ret = unsafe { ffi::nn_tensor_type(self.ptr) };
        return TensorType::try_from(ret).unwrap();
    }

    pub fn engine(&self) -> Option<&Engine> {
        let ret = unsafe { ffi::nn_tensor_engine(self.ptr) };
        if ret.is_null() {
            return None;
        }
        let engine = Engine::wrap(ret).unwrap();
        self.engine.set(Some(engine));
        return unsafe { (&*self.engine.as_ptr()).as_ref() };
    }

    pub fn shape(&self) -> &[i32] {
        let ret = unsafe { ffi::nn_tensor_shape(self.ptr) };
        let ra = unsafe { std::slice::from_raw_parts(ret, 4) };
        return ra;
    }

    pub fn dims(&self) -> i32 {
        return unsafe { ffi::nn_tensor_dims(self.ptr) };
    }

    pub fn volume(&self) -> i32 {
        return unsafe { ffi::nn_tensor_volume(self.ptr) };
    }

    pub fn size(&self) -> i32 {
        return unsafe { ffi::nn_tensor_size(self.ptr) };
    }

    pub fn axis(&self) -> i16 {
        return unsafe { ffi::nn_tensor_axis(self.ptr) as i16 };
    }

    pub fn zeros(&self) -> Result<&[i32], Error> {
        let mut zeros: usize = 0;
        let ret = unsafe { ffi::nn_tensor_zeros(self.ptr, &mut zeros as *mut usize) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("zeros returned null")));
        }
        return unsafe { Ok(std::slice::from_raw_parts(ret, zeros)) };
    }

    pub fn set_scales(&mut self, scales: &[f32]) -> Result<(), Error> {
        self.scales = Some(scales.to_vec());
        if scales.len() < (self.axis() as usize) || scales.len() != 1 {
            return Err(Error::WrapperError(String::from(
                "scales should either have length of 1 or equal to channel_dimension (axis)",
            )));
        }
        unsafe {
            ffi::nn_tensor_set_scales(self.ptr, scales.len(), scales.as_ptr() as *const f32, 0)
        };
        return Ok(());
    }

    pub fn set_aux_object<T>(&self, aux_object: &mut T) {
        let ptr = aux_object as *mut T;
        unsafe {
            ffi::nn_tensor_set_aux_object(self.ptr, ptr as *mut std::ffi::c_void, None);
        };
    }

    fn mapro_(&self) -> Result<*const ::std::os::raw::c_void, Error> {
        let ret = unsafe { ffi::nn_tensor_mapro(self.ptr) };
        if ret.is_null() {
            return Err(Error::WrapperError("nn_tensor_mapro failed".to_string()));
        }
        return Ok(ret);
    }

    pub fn mapro<'a>(&'a self) -> Result<TensorData<'a>, Error> {
        let tensor_type = self.tensor_type();
        let size = self.size();
        match tensor_type {
            TensorType::RAW => {
                let ptr = self.mapro_()? as *const u8;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::RAW(sret),
                });
            }
            TensorType::STR => {
                let ptr = self.mapro_()? as *const i8;
                let cstr = unsafe { CStr::from_ptr(ptr) };
                let str_temp = cstr.to_str()?;
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::STR(str_temp),
                });
            }
            TensorType::I8 => {
                let ptr = self.mapro_()? as *const i8;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::I8(sret),
                });
            }
            TensorType::U8 => {
                let ptr = self.mapro_()? as *const u8;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::U8(sret),
                });
            }
            TensorType::I16 => {
                let ptr = self.mapro_()? as *const i16;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::I16(sret),
                });
            }
            TensorType::U16 => {
                let ptr = self.mapro_()? as *const u16;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::U16(sret),
                });
            }
            TensorType::I32 => {
                let ptr = self.mapro_()? as *const i32;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::I32(sret),
                });
            }
            TensorType::U32 => {
                let ptr = self.mapro_()? as *const u32;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::U32(sret),
                });
            }
            TensorType::I64 => {
                let ptr = self.mapro_()? as *const i64;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::I64(sret),
                });
            }
            TensorType::U64 => {
                let ptr = self.mapro_()? as *const u64;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::U64(sret),
                });
            }
            TensorType::F16 => {
                let ptr = self.mapro_()? as *const u8;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::RAW(sret),
                });
            }
            TensorType::F32 => {
                let ptr = self.mapro_()? as *const f32;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::F32(sret),
                });
            }
            TensorType::F64 => {
                let ptr = self.mapro_()? as *const f64;
                let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
                return Ok(TensorData {
                    tensor: self,
                    data: MappedData::F64(sret),
                });
            }
        }
    }

    unsafe fn unmap(&self) {
        unsafe { ffi::nn_tensor_unmap(self.ptr) };
    }

    pub unsafe fn from_ptr(ptr: *mut ffi::NNTensor, owned: bool) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from("ptr is null")));
        }

        return Ok(Tensor {
            owned,
            engine: Cell::new(None),
            ptr,
            scales: None,
        });
    }

    pub fn to_mut_ptr(&self) -> *mut ffi::NNTensor {
        return self.ptr;
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                ffi::nn_tensor_release(self.ptr);
            };
        }
    }
}
