use std::collections::HashMap;
use thiserror::Error;
use crate::ast::*;

#[derive(Debug, Error)]
pub enum AnalyzerError {
    #[error("Type error: {0}")]
    TypeError(String),
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),
    #[error("Undefined function: {0}")]
    UndefinedFunction(String),
    #[error("Undefined type: {0}")]
    UndefinedType(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub ty: AstType,
    pub mutable: bool,
}

pub struct Analyzer {
    // Track variable types in each scope
    scopes: Vec<HashMap<String, TypeInfo>>,
    // Track function signatures
    functions: HashMap<String, (Vec<AstType>, Option<AstType>)>,
    // Track struct definitions
    structs: HashMap<String, Vec<(String, AstType)>>,
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            structs: HashMap::new(),
        }
    }

    pub fn analyze(&mut self, program: &Program) -> Result<(), AnalyzerError> {
        // First pass: collect all type definitions and function signatures
        self.collect_definitions(program)?;
        
        // Second pass: analyze function bodies and expressions
        self.analyze_bodies(program)?;
        
        Ok(())
    }

    fn collect_definitions(&mut self, program: &Program) -> Result<(), AnalyzerError> {
        for item in &program.items {
            match item {
                Item::Function(func) => {
                    let param_types: Vec<AstType> = func.params.iter()
                        .map(|p| p.ty.clone())
                        .collect();
                    self.functions.insert(func.name.clone(), (param_types, func.return_type.clone()));
                }
                Item::Struct(struct_def) => {
                    let fields: Vec<(String, AstType)> = struct_def.fields.iter()
                        .map(|f| (f.name.clone(), f.ty.clone()))
                        .collect();
                    self.structs.insert(struct_def.name.clone(), fields);
                }
            }
        }
        Ok(())
    }

    fn analyze_bodies(&mut self, program: &Program) -> Result<(), AnalyzerError> {
        for item in &program.items {
            if let Item::Function(func) = item {
                self.push_scope();
                
                // Add parameters to scope
                for param in &func.params {
                    self.add_variable(&param.name, TypeInfo {
                        ty: param.ty.clone(),
                        mutable: false,
                    });
                }
                
                // Analyze function body
                self.analyze_block(&func.body)?;
                
                self.pop_scope();
            }
        }
        Ok(())
    }

    fn analyze_block(&mut self, block: &Block) -> Result<AstType, AnalyzerError> {
        self.push_scope();
        
        let mut last_type = AstType::Int; // Default return type
        for stmt in &block.statements {
            last_type = self.analyze_statement(stmt)?;
        }
        
        self.pop_scope();
        Ok(last_type)
    }

    fn analyze_statement(&mut self, stmt: &Stmt) -> Result<AstType, AnalyzerError> {
        match stmt {
            Stmt::Let { name, ty, value } => {
                let value_type = if let Some(expr) = value {
                    self.analyze_expression(expr)?
                } else {
                    ty.clone().unwrap_or(AstType::Int)
                };
                
                if let Some(explicit_type) = ty {
                    if *explicit_type != value_type {
                        return Err(AnalyzerError::TypeError(format!(
                            "Type mismatch: expected {:?}, found {:?}",
                            explicit_type, value_type
                        )));
                    }
                }
                
                self.add_variable(name, TypeInfo {
                    ty: value_type.clone(),
                    mutable: true,
                });
                
                Ok(value_type)
            }
            Stmt::Expr(expr) => self.analyze_expression(expr),
            Stmt::Return(Some(expr)) => self.analyze_expression(expr),
            Stmt::Return(None) => Ok(AstType::Int), // Default return type
            _ => Ok(AstType::Int), // TODO: Implement other statements
        }
    }

    fn analyze_expression(&mut self, expr: &Expr) -> Result<AstType, AnalyzerError> {
        match expr {
            Expr::Literal(lit) => Ok(match lit {
                Literal::Int(_) => AstType::Int,
                Literal::Float(_) => AstType::Float,
                Literal::Bool(_) => AstType::Bool,
                Literal::String(_) => AstType::String,
                Literal::Char(_) => AstType::Char,
            }),
            Expr::VarRef(name) => {
                self.lookup_variable(name)
                    .map(|info| info.ty.clone())
                    .ok_or_else(|| AnalyzerError::UndefinedVariable(name.clone()))
            }
            Expr::Binary { op, lhs, rhs } => {
                let lhs_type = self.analyze_expression(lhs)?;
                let rhs_type = self.analyze_expression(rhs)?;
                
                if lhs_type != rhs_type {
                    return Err(AnalyzerError::TypeError(format!(
                        "Type mismatch in binary operation: {:?} and {:?}",
                        lhs_type, rhs_type
                    )));
                }
                
                Ok(lhs_type)
            }
            _ => Ok(AstType::Int), // TODO: Implement other expressions
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn add_variable(&mut self, name: &str, type_info: TypeInfo) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), type_info);
    }

    fn lookup_variable(&self, name: &str) -> Option<&TypeInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(info) = scope.get(name) {
                return Some(info);
            }
        }
        None
    }
} 