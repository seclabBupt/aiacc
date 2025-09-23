# Lab Project: ML Shape Computation Optimizer Pass

## 项目概述

在机器学习编译器中，大量的计算涉及张量形状的推导和算术运算。许多这些计算在编译时就能确定，通过常量折叠和形状推导可以显著提升运行时性能。本项目要求你实现一个MLIR Pass来优化这类计算。

## 学习目标

通过完成本项目，你将：
- 掌握MLIR Pass的完整开发流程
- 理解常量折叠和形状推导的编译器优化技术
- 应用所有课程中学到的MLIR Dialect知识
- 学会测试驱动的编译器开发方法
- 为TPU_MLIR等真实项目打下基础

## 项目背景

### 问题场景
在神经网络编译中，经常遇到如下代码模式：

```mlir
// 卷积层输出形状计算
%input_h = arith.constant 224 : index
%kernel = arith.constant 3 : index
%stride = arith.constant 2 : index
%output_h = arith.divsi %input_h, %stride : index  // 可以编译时计算

// 动态张量创建
%size = arith.muli %dim1, %dim2 : index  // 常量相乘
%tensor = tensor.empty(%size) : tensor<?xf32>  // 可以变为静态形状
```

这些计算完全可以在编译时完成，避免运行时开销。

### 优化目标
你的Pass需要将上述代码优化为：

```mlir
%output_h = arith.constant 112 : index  // 直接计算结果
%tensor = tensor.empty() : tensor<200xf32>  // 静态形状
```

## 技术要求

### 必须实现的优化
1. **算术常量折叠** (arith dialect)
   - 加法、减法、乘法、除法的常量操作
   - 比较操作的常量条件

2. **张量形状推导** (tensor dialect)  
   - `tensor.empty` 的静态形状转换
   - 动态维度到静态维度的推导

3. **控制流简化** (scf dialect)
   - 常量条件的 `scf.if` 分支选择
   - 死代码消除

4. **函数优化** (builtin dialect)
   - 简单的常量传播

### 知识点覆盖

| 课程Lab | 在项目中的应用 | 具体技术 |
|---------|---------------|----------|
| Lab 1-2 | Pass基础架构 | PassWrapper, 操作遍历 |
| Lab 4 | Builtin操作 | 模块遍历, 函数处理 |
| Lab 5 | Arith优化 | 常量折叠, 算术简化 |
| Lab 6 | SCF简化 | 条件分支优化 |
| Lab 7 | Memory相关 | 静态大小优化建议 |
| Lab 8 | Tensor操作 | 形状推导, 静态化 |
| Lab 9 | Linalg操作 | 特殊矩阵操作识别 |
| Lab 10 | Pass系统 | RewritePattern |

## 项目结构

```
shape-optimizer-project/
├── test-cases/
│   ├── input/           # 优化前的测试用例
│   ├── expected/        # 期望的优化结果
│   └── run-tests.sh     # 自动化测试脚本
├── framework/
│   ├── CMakeLists.txt   # 构建配置
│   ├── main.cpp         # 工具主程序
│   └── PassSkeleton.cpp # Pass实现骨架
├── README.md            # 项目说明
└── 本文档.md            # 详细指导
```

## 开发步骤

### Phase 1: 环境搭建
1. 编译提供的框架代码
2. 运行测试脚本，查看初始状态
3. 理解Pass的基本结构

### Phase 2: 算术折叠
1. 实现 `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi` 的常量折叠
2. 通过 `conv_shape.mlir` 和 `matrix_chain.mlir` 测试
3. 确保基本的算术优化工作正常

### Phase 3: 形状推导
1. 实现 `tensor.empty` 的静态形状转换
2. 处理常量维度参数
3. 通过 `dynamic_shape.mlir` 测试

### Phase 4: 控制流优化
1. 实现 `scf.if` 的常量条件优化
2. 处理分支选择和死代码消除
3. 通过 `conditional_shape.mlir` 测试

### Phase 5: 集成测试
1. 运行完整测试套件
2. 修复发现的问题
3. 优化代码质量

## 测试用例说明

### Test Case 1: conv_shape.mlir
**目标**: 卷积层输出形状的编译时计算
**涉及**: arith dialect的复杂算术表达式
**难度**: ⭐⭐

### Test Case 2: dynamic_shape.mlir  
**目标**: 动态张量形状的静态化
**涉及**: tensor dialect + arith常量传播
**难度**: ⭐⭐⭐

### Test Case 3: conditional_shape.mlir
**目标**: 条件分支中的形状选择优化
**涉及**: scf dialect + 跨函数优化
**难度**: ⭐⭐⭐⭐

### Test Case 4: matrix_chain.mlir
**目标**: 矩阵操作链的形状和初始化优化
**涉及**: linalg dialect + 综合优化
**难度**: ⭐⭐

## 实现提示

### 核心算法思路
```cpp
void runOnOperation() override {
  // 1. 收集所有常量定义
  DenseMap<Value, APInt> constantValues;
  collectConstants(constantValues);
  
  // 2. 遍历所有算术操作，尝试折叠
  walkArithmeticOps(constantValues);
  
  // 3. 推导张量的静态形状
  propagateStaticShapes(constantValues);
  
  // 4. 简化控制流
  simplifyControlFlow(constantValues);
  
  // 5. 清理死代码
  eliminateDeadCode();
}
```

### 关键技术点
1. **常量识别**: 使用 `dyn_cast<arith::ConstantOp>()` 识别常量
2. **值替换**: 使用 `Value::replaceAllUsesWith()` 更新使用关系
3. **操作删除**: 使用 `Operation::erase()` 删除无用操作
4. **类型构造**: 使用 `RankedTensorType::get()` 创建静态形状类型

### 常见陷阱
- 不要在遍历过程中直接修改IR，先收集再批量处理
- 注意操作的使用计数，避免删除仍被使用的操作
- 确保类型兼容性，静态形状转换要保持元素类型不变

## 提交要求

### 必需文件
- 修改后的 `PassSkeleton.cpp` (重命名为你的实现)
- 修改后的 `CMakeLists.txt` (如有需要)
- 测试运行报告 (test_results.txt)
- 项目总结文档 (summary.md)

### 可选文件
- 额外的测试用例
- 性能分析报告

## 参考资源

- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [MLIR Operation Definition](https://mlir.llvm.org/docs/LangRef/#operations)
- [Constant Folding in Compilers](https://en.wikipedia.org/wiki/Constant_folding)
- 课程中的所有Lab实验代码

## 技术支持

如遇到问题，可以参考：
1. 运行 `./shape-opt --help` 查看工具使用方法
2. 使用 `mlir-opt --verify-diagnostics` 验证MLIR语法
3. 参考课程Lab中的类似实现模式
4. 查看MLIR官方文档和示例代码

祝你编程愉快！🚀