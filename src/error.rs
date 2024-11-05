use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error("Unable to JIT compile `sum`")]
    JitCompile,
    #[error("Unsupported type")]
    UnsupportedType,
    
    #[error("Function not found")]
    FunctionNotFound,
    
    #[error("Variable not found")]
    VariableNotFound,
    
    #[error("Invalid condition")]
    InvalidCondition,
    
    #[error("Break statement outside loop")]
    BreakOutsideLoop,
    
    #[error("Continue statement outside loop")]
    ContinueOutsideLoop,
    
    #[error("Unsupported expression")]
    UnsupportedExpression,
    
    #[error("Feature not implemented")]
    Unimplemented,
}
