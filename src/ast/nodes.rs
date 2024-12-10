/// The root node of our AST, representing a complete program
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

/// Top-level items in our program (functions, structs, etc.)
#[derive(Debug, Clone)]
pub enum Item {
    Function(ItemFunction),
    Struct(ItemStruct),
    // TODO: Add more items like Impl, Trait, etc.
}

/// A function definition
#[derive(Debug, Clone)]
pub struct ItemFunction {
    pub name: String,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<AstType>,
    pub body: Block,
}

/// A function parameter
#[derive(Debug, Clone)]
pub struct FunctionParam {
    pub name: String,
    pub ty: AstType,
}

/// A struct definition
#[derive(Debug, Clone)]
pub struct ItemStruct {
    pub name: String,
    pub fields: Vec<StructField>,
}

/// A struct field
#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub ty: AstType,
}

/// A block of statements
#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Stmt>,
}

/// A statement in our language
#[derive(Debug, Clone)]
pub enum Stmt {
    /// Let binding: let name: type = value;
    Let {
        name: String,
        ty: Option<AstType>,
        value: Option<Expr>,
    },
    /// Expression statement (without semicolon)
    Expr(Expr),
    /// Expression with semicolon
    Semi(Expr),
    /// Return statement
    Return(Option<Expr>),
    /// While loop
    While {
        condition: Expr,
        body: Block,
    },
    /// For loop
    For {
        var: String,
        iterator: Expr,
        body: Block,
    },
}

/// An expression in our language
#[derive(Debug, Clone)]
pub enum Expr {
    /// Literal value (integer, boolean, etc.)
    Literal(Literal),
    /// Variable reference
    VarRef(String),
    /// Binary operation
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Unary operation
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// If expression
    If {
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Block>,
    },
    /// Function call
    Call {
        func: String,
        args: Vec<Expr>,
    },
    /// Field access (e.g., point.x)
    FieldAccess {
        expr: Box<Expr>,
        field: String,
    },
    /// Array indexing
    Index {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    /// Array literal
    Array(Vec<Expr>),
}

/// Literal values
#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Char(char),
}

/// Types in our language
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AstType {
    Int,
    Float,
    Bool,
    String,
    Char,
    Array(Box<AstType>),
    Struct(String),
    Function {
        params: Vec<AstType>,
        return_type: Box<AstType>,
    },
    Reference(Box<AstType>),
    Optional(Box<AstType>),
}

/// Binary operators
#[derive(Debug, Clone)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Neg,    // -
    Not,    // !
    Deref,  // *
    Ref,    // &
} 