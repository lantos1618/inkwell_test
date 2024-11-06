use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
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
