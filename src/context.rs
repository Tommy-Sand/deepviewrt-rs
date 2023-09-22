use crate::{engine::Engine, error::Error, model::Model, tensor::Tensor};
use deepviewrt_sys as ffi;
use std::{cell::Cell, ffi::CString, ptr};

pub struct Context {
    owned: bool,
    ptr: *mut ffi::NNContext,
    engine: Cell<Option<Engine>>,
    model_data: Option<Vec<u8>>,
    model: Cell<Option<Model>>,
    //	tensors: Cell
    //	tensor_ref: Vec<(i32, Tensor)>
}

impl Context {
    pub fn sizeof() -> usize {
        return unsafe { ffi::nn_context_sizeof() };
    }

    pub fn new(engine: Engine, memory_size: usize, cache_size: usize) -> Result<Context, Error> {
        let ret = unsafe {
            ffi::nn_context_init(
                engine.to_ptr_mut(),
                memory_size,
                ptr::null_mut(),
                cache_size,
                ptr::null_mut(),
            )
        };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from(
                "nn_context_init returned null",
            )));
        }
        Ok(Context {
            owned: true,
            ptr: ret,
            engine: Cell::new(Some(engine)),
            model_data: None,
            model: Cell::new(None),
        })
    }

    /*
    pub fn cache(&self) -> Option<Tensor> {

    }

    pub fn mempool(&self) -> Option<Tensor> {

    }
    */

    pub fn engine(&self) -> Option<&Engine> {
        let engine_ptr = self.engine.as_ptr();
        if !engine_ptr.is_null() {
            let engine_ref = unsafe { &*engine_ptr };
            if engine_ref.is_some() {
                return engine_ref.as_ref();
            }
        }

        let ret = unsafe { ffi::nn_context_engine(self.ptr) };
        if ret.is_null() {
            return None;
        }
        let engine = Engine::wrap(ret).unwrap();
        self.engine.set(Some(engine));
        return unsafe { (&*self.engine.as_ptr()).as_ref() };
    }

    pub fn model(&self) -> Option<&Model> {
        if !self.model.as_ptr().is_null() {
            let model_ref = unsafe { &*self.model.as_ptr() };
            if model_ref.is_some() {
                return model_ref.as_ref();
            }
        }

        let ret = unsafe { ffi::nn_context_model(self.ptr) };
        if ret.is_null() {
            return None;
        }
        let model = match unsafe { Model::try_from_ptr(ret) } {
            Ok(model) => Some(model),
            Err(e) => {
                eprintln!("{}", e);
                None
            }
        };
        self.model.set(model);

        let model_ref = unsafe { &*self.model.as_ptr() };
        if model_ref.is_some() {
            return model_ref.as_ref();
        }
        None
    }

    pub fn load_model(&mut self, data: Vec<u8>) -> Result<(), Error> {
        self.unload_model();
        //Insert and get the mode_data reference
        let model_data_ref = self.model_data.insert(data);
        let ret = unsafe {
            ffi::nn_context_model_load(
                self.ptr as *mut ffi::NNContext,
                model_data_ref.len(),
                model_data_ref.as_ptr() as *const std::ffi::c_void,
            )
        };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }
        return Ok(());
    }

    pub fn unload_model(&mut self) {
        unsafe { ffi::nn_context_model_unload(self.ptr) };
        self.model_data = None;
        self.model.set(None);
    }

    pub fn run_model(&self) {}

    pub fn tensor(&self, name: &str) -> Option<Tensor> {
        let cname = match CString::new(name) {
            Ok(cname) => cname,
            Err(_) => return None,
        };

        let ret = unsafe { ffi::nn_context_tensor(self.ptr, cname.into_raw()) };
        if ret.is_null() {
            return None;
        }
        let tensor = unsafe { Tensor::from_ptr(ret, false).unwrap() };
        return Some(tensor);
    }

    pub fn tensor_index(&self, index: usize) -> Option<Tensor> {
        let ret = unsafe { ffi::nn_context_tensor_index(self.ptr, index) };
        if ret.is_null() {
            return None;
        }
        let tensor = unsafe { Tensor::from_ptr(ret, false).unwrap() };
        return Some(tensor);
    }

    pub unsafe fn from_ptr(ptr: *mut ffi::NNContext) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from("ptr is null")));
        }

        return Ok(Self {
            owned: false,
            ptr,
            engine: Cell::new(None),
            model_data: None,
            model: Cell::new(None),
        });
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.owned {
            unsafe { ffi::nn_context_release(self.ptr) };
        }
    }
}
