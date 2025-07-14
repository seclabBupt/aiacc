# MLIR Dialect 入门学习笔记

## 1. MLIR 和 Dialect 简介

### 1.1 什么是 MLIR？
MLIR（Multi-Level Intermediate Representation）是 LLVM 生态系统中的一个编译器基础设施，旨在提供一种灵活、可扩展的中间表示（IR），以支持多种编程模型和硬件目标。MLIR 的设计目标是：
- **多层次表示**：支持从高级抽象（如 TensorFlow 图）到低级硬件指令的表示。
- **可扩展性**：通过 Dialect 机制，允许用户定义特定领域的操作和类型。
- **优化能力**：提供强大的变换（Pass）机制，用于优化和代码生成。

### 1.2 什么是 Dialect？
Dialect 是 MLIR 的核心模块化机制，代表一组特定领域的操作（Operations）、类型（Types）和属性（Attributes）。每个 Dialect 针对特定的计算模型或硬件目标，例如：
- **机器学习**：`linalg` Dialect 用于线性代数操作。
- **硬件加速**：`gpu` 或 `spirv` Dialect 用于 GPU 或 Vulkan。
- **通用计算**：`std` Dialect 提供通用操作，如算术和控制流。

Dialect 的优点：
- 模块化：不同 Dialect 相互独立，互不干扰。
- 可组合：可以在同一 IR 中混合多个 Dialect。
- 可转换：通过 Pass 将一个 Dialect 的操作转换为另一个 Dialect。

### 1.3 常见 Dialect 概览
以下是一些内置 Dialect 的简要介绍：
- **`std`**：标准 Dialect，包含通用操作，如 `addf`（浮点加法）、`br`（分支）、`func`（函数定义）。
- **`affine`**：用于表达仿射变换（Affine Transformations），适合循环优化和内存访问分析。
- **`linalg`**：提供高级线性代数操作，如矩阵乘法、卷积，广泛用于机器学习框架。
- **`memref`**：管理内存分配和访问，适合低级优化。
- **`gpu`**：针对 GPU 并行计算，提供线程管理和内存操作。
- **`vector`**：支持向量化和 SIMD 指令。

## 2. Dialect 的核心组件

### 2.1 操作（Operations）
操作是 MLIR 的基本执行单元，类似于函数调用或指令。每个操作属于某个 Dialect，并定义了：
- **输入**：操作的输入值（可以是值或张量）。
- **输出**：操作的返回值。
- **属性**：操作的静态元数据（如常量值）。
- **行为**：操作的语义，通常在 C++ 中实现。

**TableGen 定义示例**：
```tablegen
def MyAddOp : Op<MyDialect, "add"> {
  let summary = "Floating point addition operation";
  let arguments = (ins F32:$lhs, F32:$rhs);  // 两个 f32 输入
  let results = (outs F32:$result);          // 一个 f32 输出
  let assemblyFormat = "$lhs, $rhs attr-dict"; // 自定义 IR 打印格式
}
```
上述代码定义了一个浮点加法操作 `mydialect.add`，接受两个 `f32` 输入，产生一个 `f32` 输出。

**IR 示例**：
```mlir
%0 = mydialect.add %a, %b : f32
```

### 2.2 类型（Types）
MLIR 支持内置类型（如 `i32`、`f64`、`memref`）和自定义类型。自定义类型通过 Dialect 定义，适合特定领域的数据结构。

**TableGen 定义示例**：
```tablegen
def MyMatrixType : TypeDef<MyDialect, "matrix"> {
  let summary = "A matrix type with dynamic dimensions";
  let parameters = (type F32, I64:$rows, I64:$cols); // 浮点元素，行列数
}
```
上述代码定义了一个矩阵类型，包含浮点元素和动态的行/列数。

**IR 示例**：
```mlir
%0 = mydialect.create_matrix 2, 3 : !mydialect.matrix<f32, 2, 3>
```

### 2.3 属性（Attributes）
属性是操作的静态元数据，用于存储常量、配置或优化信息。属性可以是标量（如整数、浮点数）、数组或字典。

**TableGen 定义示例**：
```tablegen
def MyOp : Op<MyDialect, "my_op"> {
  let summary = "Custom operation with attributes";
  let arguments = (ins F32:$input);
  let results = (outs F32:$output);
  let attributes = (ins BoolAttr:$isFast, I32Attr:$priority);
}
```
上述操作包含两个属性：`isFast`（布尔值）和 `priority`（整数）。

**IR 示例**：
```mlir
%0 = mydialect.my_op %input {isFast = true, priority = 42} : f32 -> f32
```

### 2.4 区域（Regions）
一些操作支持区域（Regions），即嵌套的 IR 块，用于表达控制流或复杂逻辑。例如，`func` 操作包含一个区域来定义函数体。

**示例**：
```mlir
func @main() {
  %0 = mydialect.add %a, %b : f32
  return
}
```

## 3. 创建自定义 Dialect

### 3.1 定义 Dialect
使用 TableGen 定义 Dialect 的基本信息：
```tablegen
def MyDialect : Dialect {
  let name = "mydialect";                     // Dialect 名称
  let summary = "A custom dialect for learning MLIR";
  let cppNamespace = "mydialect";             // C++ 命名空间
  let dependentDialects = ["std"];            // 依赖标准 Dialect
}
```

### 3.2 定义操作
在 TableGen 中定义操作的接口：
```tablegen
def MyMulOp : Op<MyDialect, "mul"> {
  let summary = "Floating point multiplication operation";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs, $rhs attr-dict";
}
```

### 3.3 实现 Dialect 和操作（C++）
在 C++ 中实现 Dialect 和操作的具体逻辑。

**头文件（MyDialect.h）**：
```cpp
#ifndef MYDIALECT_H
#define MYDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace mydialect {

class MyDialect : public Dialect {
public:
  explicit MyDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "mydialect"; }
  void initialize();
};

#define GET_OP_CLASSES
#include "MyDialectOps.h.inc"

} // namespace mydialect
} // namespace mlir

#endif // MYDIALECT_H
```

**实现文件（MyDialect.cpp）**：
```cpp
#include "MyDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mydialect;

MyDialect::MyDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MyDialect>()) {
  initialize();
}

void MyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MyDialectOps.cpp.inc"
      >();
}
```

**操作接口（MyDialectOps.td）**：
```tablegen
include "MyDialect.h"

def MyMulOp : Op<MyDialect, "mul"> {
  let summary = "Floating point multiplication operation";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
}
```

**操作实现**：
在 C++ 中为操作实现验证逻辑或语义：
```cpp
void MyMulOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
  state.addOperands({lhs, rhs});
  state.addTypes(builder.getF32Type());
}

LogicalResult MyMulOp::verify() {
  if (!getLhs().getType().isa<FloatType>() || !getRhs().getType().isa<FloatType>())
    return emitOpError("requires floating-point operands");
  return success();
}
```

### 3.4 注册 Dialect
在 MLIR 上下文中注册 Dialect：
```cpp
void registerMyDialect(MLIRContext *context) {
  context->getOrLoadDialect<mydialect::MyDialect>();
}
```

### 3.5 测试 Dialect
使用 MLIR 工具链测试：
- **生成 IR**：
  ```mlir
  func @main() {
    %a = constant 2.0 : f32
    %b = constant 3.0 : f32
    %c = mydialect.mul %a, %b : f32
    return
  }
  ```
- **使用 mlir-opt**：
  ```bash
  mlir-opt -allow-unregistered-dialect input.mlir -o output.mlir
  ```
- **验证**：使用 `FileCheck` 编写测试用例，确保 IR 输出正确。

## 4. Dialect 的进阶功能

### 4.1 转换 Pass
Dialect 通常需要通过 Pass 转换为其他 Dialect 或优化。例如，将 `mydialect.mul` 转换为 `std.mulf`：
```cpp
class MyMulToStdPass : public PassWrapper<MyMulToStdPass, OperationPass<>> {
  void runOnOperation() override {
    auto *ctx = getOperation().getContext();
    ConversionTarget target(*ctx);
    target.addIllegalOp<mydialect::MyMulOp>();
    target.addLegalDialect<StandardOpsDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add([](MyMulOp op, PatternRewriter &rewriter) {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), op.getRhs());
      return success();
    });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
```

### 4.2 自定义类型解析
为自定义类型实现解析和打印逻辑：
```cpp
class MyMatrixType : public Type::TypeBase<MyMatrixType, Type, TypeStorage> {
public:
  using Base::Base;
  static MyMatrixType get(MLIRContext *context, Type elementType, int64_t rows, int64_t cols) {
    return Base::get(context, elementType, rows, cols);
  }
};
```

### 4.3 属性验证
为属性添加约束，例如限制 `priority` 属性为正整数：
```tablegen
def PositiveIntAttr : I32Attr {
  let constraints = [CArg<I32, "value > 0">];
}
```