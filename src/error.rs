use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error("Unable to JIT compile `sum`")]
    JitCompile,
}
