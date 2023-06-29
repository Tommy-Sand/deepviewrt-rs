#!/bin/sh

bindgen --allowlist-function 'nn_.*' deepview-rt.h > src/ffi.rs
