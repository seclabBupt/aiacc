
# MLIR Dialect 学习笔记 模块3 优化

## 1. 优化的核心概念

**优化是什么？**  
优化就是改 IR（中间表示），让程序跑得更快、更省资源。IR 是程序的“半成品代码”，优化就像修剪多余枝叶，让它更高效。  
比如：
- **常量折叠**：`x + 0` 直接变成 `x`，因为加 0 没用。
- **死代码消除**：删掉没人用的变量或代码。
- **循环展开/函数内联**：把循环或函数调用展开，减少开销。

**Pass 是什么？**  
Pass 是 MLIR 里的“优化工具”，每次优化都通过一个 Pass 来完成。Pass 就像一个工人，拿着具体任务（比如删掉多余代码）去改造 IR。

**例子：**
```mlir
%0 = arith.constant 0 : i32
%1 = arith.addi %x, %0 : i32
```
优化后：
```mlir
%1 = %x
```
解释：加 0 没意义，直接用 `x` 替换，省掉无用操作。

---

## 2. MLIR 内置的优化 Pass

MLIR 自带了一些强大的 Pass，像工具箱里的常用工具，帮你快速优化 IR。以下是几个主要的：

| Pass 名称       | 通俗解释                                      |
|----------------|---------------------------------------------|
| `canonicalize` | 把 IR 整理成标准、简洁的格式，删掉多余逻辑，像是代码大扫除。 |
| `cse`          | 找到重复的计算，合并成一个，省时间和空间。 |
| `inline`       | 把函数调用替换成函数内容，省去调用开销。 |
| `symbol-dce`   | 删除没用的函数或变量，清理垃圾代码。 |
| `loop-unroll`  | 把循环展开成多行代码，减少循环的开销。 |

### 2.1 Canonicalize
**作用**：把 IR 改成更简洁的“标准形式”，去掉多余操作。  
**例子：**
```mlir
%0 = arith.constant 0 : i32
%1 = arith.addi %x, %0 : i32
```
优化后：`%1 = %x`，直接去掉加 0 的操作。

### 2.2 CSE（公共子表达式消除）
**作用**：发现重复的计算，只算一次，节省资源。  
**例子：**
```mlir
%1 = arith.addi %a, %b : i32
%2 = arith.addi %a, %b : i32
```
优化后：
```mlir
%1 = arith.addi %a, %b : i32
%2 = %1
```
解释：两次一样的加法合并，第二次直接用第一次的结果。

### 2.3 Inlining（函数内联）
**作用**：把函数调用替换成函数的具体代码，减少调用开销。  
**例子：**
```mlir
func.func @add(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}
%1 = func.call @add(%x, %y) : (i32, i32) -> i32
```
优化后：
```mlir
%0 = arith.addi %x, %y : i32
%1 = %0
```
解释：函数调用被替换成加法操作，省去调用过程。

### 2.4 运行 Pass 的方法
用 `mlir-opt` 工具运行优化：
```bash
mlir-opt input.mlir -canonicalize -cse -o output.mlir
```
解释：对 `input.mlir` 先标准化（`canonicalize`），再消除重复表达式（`cse`），结果保存到 `output.mlir`。

---

## 3. 为什么需要自定义 Pass？

MLIR 的内置 Pass（如 `cse`、`canonicalize`）只认识标准 Dialect（比如 `arith`、`func`）。如果你定义了自己的 Dialect（比如 `mydialect`），这些 Pass 就不认识你的操作，无法优化。

**例子：**
```mlir
%1 = mydialect.addi %a, %b
%2 = mydialect.addi %a, %b
```
`cse` 不认识 `mydialect.addi`，不会合并重复的加法。所以，你得自己写一个 Pass，专门处理你的 Dialect。

---

## 4. 编写自定义 Pass

我们以优化 `mydialect.addi` 为例，目标是把 `x + 0` 或 `0 + x` 简化为 `x`。

### 4.1 测试用 IR
```mlir
module {
  func.func @test(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %sum = mydialect.addi %arg0, %c0 : i32
    return %sum : i32
  }
}
```
目标：把 `mydialect.addi %arg0, %c0` 优化成 `%sum = %arg0`。

### 4.2 编写 Pass
1. **定义 Pass**：创建一个继承 `PassWrapper` 的 C++ 类，指定作用在函数上（`OperationPass<func::FuncOp>`）。
2. **实现逻辑**：在 `runOnOperation()` 方法里遍历 IR，找到 `mydialect.addi`，检查是否加 0，然后替换。
3. **注册 Pass**：让 MLIR 认识你的 Pass。

**代码示例：**

**MyPass.h**
```c
#ifndef MY_PASS_H
#define MY_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<OperationPass<func::FuncOp>> createRemoveMyAddZeroPass();
}

#endif
```

**MyPass.cpp**
```c
#include "MyPass.h"
#include "mydialect/MyOps.h.inc"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct RemoveMyAddZeroPass
    : public PassWrapper<RemoveMyAddZeroPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    func.walk([&](mydialect::AddIOp op) {
      Value lhs = op.getLhs();
      Value rhs = op.getRhs();

      auto isZeroConst = [](Value val) -> bool {
        if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
          if (auto attr = cst.getValue().dyn_cast<IntegerAttr>())
            return attr.getValue().isZero();
        }
        return false;
      };

      if (isZeroConst(lhs)) {
        op.replaceAllUsesWith(rhs);
        op.erase();
      } else if (isZeroConst(rhs)) {
        op.replaceAllUsesWith(lhs);
        op.erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createRemoveMyAddZeroPass() {
  return std::make_unique<RemoveMyAddZeroPass>();
}
```

**解释**：
- `runOnOperation()` 遍历函数中的 `mydialect.addi` 操作。
- 用 `isZeroConst` 检查操作数是否为 0。
- 如果左边（`lhs`）或右边（`rhs`）是 0，就把整个操作替换为另一个操作数，并删除原操作。

### 4.3 注册和运行
**主程序：**
```c
#include "mlir/Support/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mydialect/MyDialect.h"
#include "MyPass.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::mydialect::MyDialect>();
  mlir::registerAllPasses();
  mlir::PassRegistration<mlir::RemoveMyAddZeroPass>();
  return mlir::MlirOptMain(argc, argv, "My Pass Tool\n", registry);
}
```

**运行：**
```bash
./bin/my-opt test/test.mlir -remove-my-add-zero
```
结果：`mydialect.addi %arg0, %c0` 被优化为 `%sum = %arg0`。

---

## 5. Rewrite Pattern：更优雅的优化方式

Rewrite Pattern 是 MLIR 推荐的优化方法，比手写 Pass 更简单。它通过“模式匹配”自动找到需要优化的 IR 部分并替换，像是用“查找替换”工具改代码。

### 5.1 核心组成
- **OpRewritePattern<T>**：指定匹配的 Op（比如 `mydialect.addi`）。
- **matchAndRewrite()**：定义匹配条件和替换逻辑。
- **PatternRewriter**：自动处理替换和 IR 更新。
- **applyPatternsAndFoldGreedily()**：批量应用所有 Pattern。

### 5.2 编写 Pattern
目标：用 Rewrite Pattern 优化 `mydialect.addi x, 0` 为 `x`。

**MyPatterns.cpp**
```c
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mydialect/MyOps.h.inc"

using namespace mlir;

struct RemoveAddZeroPattern : OpRewritePattern<mydialect::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mydialect::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    auto isZeroConst = [](Value val) -> bool {
      if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = cst.getValue().dyn_cast<IntegerAttr>())
          return attr.getValue().isZero();
      }
      return false;
    };

    if (isZeroConst(rhs)) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    if (isZeroConst(lhs)) {
      rewriter.replaceOp(op, rhs);
      return success();
    }

    return failure();
  }
};
```

**解释**：
- `RemoveAddZeroPattern` 匹配 `mydialect.addi`。
- `matchAndRewrite` 检查操作数是否为 0，如果是，就用 `rewriter.replaceOp` 替换为非 0 操作数。
- 返回 `success()` 表示优化成功，`failure()` 表示没匹配到。

**注册 Pattern（MyPass.cpp）**
```c
#include "MyPass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mydialect/MyOps.h.inc"

namespace mlir {
namespace {

struct RemoveMyAddZeroPass
    : public PassWrapper<RemoveMyAddZeroPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<RemoveAddZeroPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRemoveMyAddZeroPass() {
  return std::make_unique<RemoveMyAddZeroPass>();
}
}
```

**解释**：
- 用 `RewritePatternSet` 注册 `RemoveAddZeroPattern`。
- `applyPatternsAndFoldGreedily` 自动应用 Pattern，优化 IR。

### 5.3 为什么用 Rewrite Pattern？
- **更简单**：不用手动遍历 IR，MLIR 自动匹配和替换。
- **更模块化**：可以定义多个 Pattern，组合使用，代码更清晰。
- **更高效**：MLIR 优化了匹配过程，性能更好。


## 版本更新记录
| 更新人 | 更新内容 | 版本号 |
|-------|:------:|------:|
| Henry Li | 模块三基本框架搭建 | v1.0.0 |