# Inkwell Test - A Rust-like Language Compiler

This project implements a compiler for a Rust-like programming language using LLVM through the Inkwell bindings. The compiler demonstrates how to generate LLVM IR from a custom AST.

## Features

### Current Implementation
- Function definitions with parameters and return types
- Basic types: Int, Float, Bool, String, Char
- Compound types: Arrays, Structs, References, Optionals
- Control flow: If expressions, While loops, For loops
- Variable declarations and assignments
- Binary and unary operations
- Function calls
- Field access for structs
- Array indexing and literals

### AST Structure
- Program: Collection of top-level items
- Items: Functions and Structs
- Statements: Let bindings, Expressions, Return, While, For
- Expressions: Literals, Variables, Operations, Control Flow
- Types: Basic types and compound types

## Code Generation Process

### Type System
The compiler maps our high-level types to LLVM types:
- `Int` → `i64`
- `Float` → `f64`
- `Bool` → `i1`
- `String` → `i8*` (pointer to null-terminated array)
- `Char` → `i8`
- `Array(T)` → `[T x 0]` (zero-sized array, will be resized at runtime)
- `Struct` → LLVM struct type with named fields
- `Function` → Function pointer type
- `Reference` → Pointer type
- `Optional` → Struct containing value and boolean flag

### Expression Generation
Key expression translations:

1. **Literals**:
```rust
// Source: 42
// LLVM IR: i64 42
ctx.context.i64_type().const_int(42, true)

// Source: "hello"
// LLVM IR: global [6 x i8] c"hello\00"
ctx.context.const_string(s.as_bytes(), true)
```

2. **Binary Operations**:
```rust
// Source: a + b
// LLVM IR: %tmp = add i64 %a, %b
ctx.builder.build_int_add(a, b, "tmp")
```

3. **If Expressions**:
```rust
// Source: if cond { then_expr } else { else_expr }
// LLVM IR:
//   %cond = icmp ne i1 %cond_val, false
//   br i1 %cond, label %then, label %else
//   then:
//     %then_val = ...
//     br label %merge
//   else:
//     %else_val = ...
//     br label %merge
//   merge:
//     %result = phi i64 [%then_val, %then], [%else_val, %else]
```

### Function Generation
Functions are generated in several steps:

1. **Function Type**:
```rust
// Source: fn add(x: i64, y: i64) -> i64
// LLVM IR: define i64 @add(i64 %x, i64 %y)
let fn_type = ret_type.fn_type(&param_types, false);
let function = module.add_function("add", fn_type, None);
```

2. **Parameter Allocation**:
```rust
// LLVM IR:
//   %x.addr = alloca i64
//   store i64 %x, i64* %x.addr
let alloc = builder.build_alloca(param_type, "x.addr");
builder.build_store(alloc, param_value);
```

3. **Return Value**:
```rust
// Source: return x + y;
// LLVM IR:
//   %tmp = add i64 %x, %y
//   ret i64 %tmp
builder.build_return(Some(&return_value));
```

### Memory Management
- Local variables are allocated on the stack using `alloca`
- String literals are stored as global constants
- Struct fields are accessed using `getelementptr`
- References are implemented as pointers

## Project Structure

```
src/
├── lib.rs           # Library entry point
├── ast/            # Abstract Syntax Tree definitions
│   ├── mod.rs      # AST module exports
│   └── nodes.rs    # AST node definitions
└── llvm_codegen/   # LLVM code generation
    ├── mod.rs      # Codegen module exports
    ├── context.rs  # Codegen context management
    ├── gen_type.rs # Type generation
    ├── gen_item.rs # Item (function/struct) generation
    ├── gen_expr.rs # Expression generation
    └── gen_stmt.rs # Statement generation
```

## Dependencies

```toml
[dependencies]
inkwell = { version = "0.5.0", features = ["llvm18-0"] }
anyhow = "1.0"
thiserror = "1.0"
```

## Usage

1. Add to your Cargo.toml:
```toml
[dependencies]
inkwell_test = { path = "path/to/inkwell_test" }
```

2. Use in your code:
```rust
use inkwell_test::{ast::*, llvm_codegen::*};

fn main() {
    let context = inkwell::context::Context::create();
    let mut codegen_ctx = CodegenContext::new(&context, "my_module");

    // Create your AST
    let program = Program {
        items: vec![
            // Your program items here
        ],
    };

    // Generate LLVM IR
    program.codegen(&mut codegen_ctx);

    // Print or use the generated IR
    println!("{}", codegen_ctx.module.print_to_string().to_string());
}
```

## Examples

### Function Definition
```rust
let function = ItemFunction {
    name: "add".to_string(),
    params: vec![
        FunctionParam {
            name: "x".to_string(),
            ty: Type::Int,
        },
        FunctionParam {
            name: "y".to_string(),
            ty: Type::Int,
        },
    ],
    return_type: Some(Type::Int),
    body: Block {
        statements: vec![
            Stmt::Return(Some(Expr::Binary {
                op: BinOp::Add,
                lhs: Box::new(Expr::VarRef("x".to_string())),
                rhs: Box::new(Expr::VarRef("y".to_string())),
            })),
        ],
    },
};
```

### If Expression
```rust
let if_expr = Expr::If {
    condition: Box::new(Expr::Binary {
        op: BinOp::Eq,
        lhs: Box::new(Expr::VarRef("x".to_string())),
        rhs: Box::new(Expr::Literal(Literal::Int(0))),
    }),
    then_branch: Block {
        statements: vec![
            Stmt::Expr(Expr::Literal(Literal::Int(42))),
        ],
    },
    else_branch: Some(Block {
        statements: vec![
            Stmt::Expr(Expr::Literal(Literal::Int(24))),
        ],
    }),
};
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Inkwell](https://github.com/TheDan64/inkwell) for the LLVM bindings
- [LLVM](https://llvm.org/) for the compiler infrastructure 