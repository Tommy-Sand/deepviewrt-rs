use deepview_rt_sys as ffi;
use std::error::Error;
use std::ffi::c_void;
use std::io;
use std::ops::Deref;

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

pub enum NNError {
    Success = 0,
    Internal = 1,
    InvalidHandle = 2,
    OutOfMemory = 3,
    OutOfResources = 4,
    NotImplemented = 5,
    InvalidParameter = 6,
    TypeMismatch = 7,
    ShapeMismatch = 8,
    InvalidShape = 9,
    InvalidOrder = 10,
    InvalidAxis = 11,
    MissingResource = 12,
    InvalidEngine = 13,
    TensorNoData = 14,
    KernelMissing = 15,
    TensorTypeUnsupported = 16,
    TooManyInputs = 17,
    SystemError = 18,
    InvalidLayer = 19,
    ModelInvalid = 20,
    ModelMissing = 21,
    StringTooLarge = 22,
    InvalidQuant = 23,
    ModelGraphFailed = 24,
    GraphVerifyFailed = 25,
    UnknownError = 26,
}

impl From<ffi::NNError> for NNError {
    fn from(value: ffi::NNError) -> Self {
        match value {
            ffi::NNError_NN_SUCCESS => {
                return NNError::Success;
            }
            ffi::NNError_NN_ERROR_INTERNAL => {
                return NNError::Internal;
            }
            ffi::NNError_NN_ERROR_INVALID_HANDLE => {
                return NNError::InvalidHandle;
            }
            ffi::NNError_NN_ERROR_OUT_OF_MEMORY => {
                return NNError::OutOfMemory;
            }
            ffi::NNError_NN_ERROR_OUT_OF_RESOURCES => {
                return NNError::OutOfResources;
            }
            ffi::NNError_NN_ERROR_NOT_IMPLEMENTED => {
                return NNError::NotImplemented;
            }
            ffi::NNError_NN_ERROR_INVALID_PARAMETER => {
                return NNError::InvalidParameter;
            }
            ffi::NNError_NN_ERROR_TYPE_MISMATCH => {
                return NNError::TypeMismatch;
            }
            ffi::NNError_NN_ERROR_SHAPE_MISMATCH => {
                return NNError::ShapeMismatch;
            }
            ffi::NNError_NN_ERROR_INVALID_SHAPE => {
                return NNError::InvalidShape;
            }
            ffi::NNError_NN_ERROR_INVALID_ORDER => {
                return NNError::InvalidOrder;
            }
            ffi::NNError_NN_ERROR_INVALID_AXIS => {
                return NNError::InvalidAxis;
            }
            ffi::NNError_NN_ERROR_MISSING_RESOURCE => {
                return NNError::MissingResource;
            }
            ffi::NNError_NN_ERROR_INVALID_ENGINE => {
                return NNError::InvalidEngine;
            }
            ffi::NNError_NN_ERROR_TENSOR_NO_DATA => {
                return NNError::TensorNoData;
            }
            ffi::NNError_NN_ERROR_KERNEL_MISSING => {
                return NNError::KernelMissing;
            }
            ffi::NNError_NN_ERROR_TENSOR_TYPE_UNSUPPORTED => {
                return NNError::TensorTypeUnsupported;
            }
            ffi::NNError_NN_ERROR_TOO_MANY_INPUTS => {
                return NNError::TooManyInputs;
            }
            ffi::NNError_NN_ERROR_SYSTEM_ERROR => {
                return NNError::SystemError;
            }
            ffi::NNError_NN_ERROR_INVALID_LAYER => {
                return NNError::InvalidLayer;
            }
            ffi::NNError_NN_ERROR_MODEL_INVALID => {
                return NNError::ModelInvalid;
            }
            ffi::NNError_NN_ERROR_MODEL_MISSING => {
                return NNError::ModelMissing;
            }
            ffi::NNError_NN_ERROR_STRING_TOO_LARGE => {
                return NNError::StringTooLarge;
            }
            ffi::NNError_NN_ERROR_INVALID_QUANT => {
                return NNError::InvalidQuant;
            }
            ffi::NNError_NN_ERROR_MODEL_GRAPH_FAILED => {
                return NNError::ModelGraphFailed;
            }
            ffi::NNError_NN_ERROR_GRAPH_VERIFY_FAILED => {
                return NNError::GraphVerifyFailed;
            }
            _ => {
                return NNError::UnknownError;
            }
        }
    }
}

pub struct NNTensor {
    ptr: *mut ffi::NNTensor,
}

impl Deref for NNTensor {
    type Target = ffi::NNTensor;

    fn deref(&self) -> &Self::Target {
        return unsafe { &*(self.ptr) };
    }
}

impl NNTensor {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let ptr = unsafe {
            ffi::nn_tensor_init(
                std::ptr::null::<c_void>() as *mut c_void,
                std::ptr::null::<ffi::nn_engine>() as *mut ffi::nn_engine,
            )
        };
        if ptr.is_null() {
            let err = io::Error::last_os_error();
            return Err(Box::new(err));
        }

        return Ok(NNTensor { ptr });
    }

    pub fn alloc(
        &self,
        ttype: NNTensorType,
        n_dims: i32,
        shape: &mut [i32; 3],
    ) -> Result<NNError, NNError> {
        let ttype_c_uint = (ttype as u32) as std::os::raw::c_uint;
        let error =
            unsafe { ffi::nn_tensor_alloc(self.ptr, ttype_c_uint, n_dims, shape.as_mut_ptr()) };

        if error == ffi::NNError_NN_SUCCESS {
            return Ok(NNError::Success);
        }

        return Err(NNError::from(error));
    }

    pub fn shape(&self) -> Result<&[i32], ()> {
        let temp = unsafe { ffi::nn_tensor_shape(self.ptr) };
        if temp.is_null() {
            return Err(());
        }

        //reference array
        let ra = unsafe { std::slice::from_raw_parts(temp, 4) };
        return Ok(ra);
    }

    //TODO Use better variable names
    pub fn mapro_u8(&self, height: i32, width: i32, depth: i32) -> Result<&[u8], ()> {
        let temp = unsafe { ffi::nn_tensor_mapro(self.ptr) };
        if temp.is_null() {
            return Err(());
        }
        let temp = temp as *const u8;
        let temp2 = unsafe { std::slice::from_raw_parts(temp, (width * height * depth) as usize) };
        return Ok(temp2);
    }

    pub fn unmap(&self) {
        unsafe { ffi::nn_tensor_unmap(self.ptr) };
    }

    pub fn to_mut_ptr(&self) -> *mut ffi::NNTensor {
        return self.ptr;
    }
}
