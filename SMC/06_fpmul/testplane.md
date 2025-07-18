# FPMUL 浮点乘法器测试计划

## 1. 测试概述

### 1.1 测试目标
- 验证FPMUL模块完全符合IEEE 754标准
- 确保FP16和FP32混合精度乘法运算的正确性
- 验证所有特殊情况和边界条件的处理
- 测试舍入机制和异常处理的准确性

### 1.2 测试覆盖率目标
- **功能覆盖率**：≥95%
- **代码覆盖率**：≥90%
- **特殊值覆盖率**：100%
- **边界值覆盖率**：100%

## 2. 测试分类与用例

### 2.1 规格数乘非规格数测试 

#### 2.1.1 FP16 规格数 × 非规格数
```
测试用例分组：
Group 1: 正规格数 × 正非规格数
- 0x3C00 (1.0) × 0x0001 (最小正非规格数)
- 0x3C00 (1.0) × 0x03FF (最大正非规格数)
- 0x4000 (2.0) × 0x0200 (中等非规格数)
- 0x7BFF (最大规格数) × 0x0001 (最小非规格数)

Group 2: 正规格数 × 负非规格数
- 0x3C00 (1.0) × 0x8001 (最小负非规格数)
- 0x4000 (2.0) × 0x83FF (最大负非规格数)

Group 3: 负规格数 × 正非规格数
- 0xBC00 (-1.0) × 0x0001 (最小正非规格数)
- 0xC000 (-2.0) × 0x03FF (最大正非规格数)

Group 4: 负规格数 × 负非规格数
- 0xBC00 (-1.0) × 0x8001 (最小负非规格数)
- 0xC000 (-2.0) × 0x83FF (最大负非规格数)
```

#### 2.1.2 FP32 规格数 × 非规格数
```
测试用例分组：
Group 1: 正规格数 × 正非规格数
- 0x3F800000 (1.0f) × 0x00000001 (最小正非规格数)
- 0x3F800000 (1.0f) × 0x007FFFFF (最大正非规格数)
- 0x40000000 (2.0f) × 0x00400000 (中等非规格数)
- 0x7F7FFFFF (最大规格数) × 0x00000001

Group 2: 边界规格数 × 非规格数
- 0x00800000 (最小正规格数) × 0x007FFFFF (最大非规格数)
- 0x80800000 (最小负规格数) × 0x80000001 (最小负非规格数)
```

#### 2.1.3 非规格数 × 规格数 (对称测试)
```
验证乘法交换律：A × B = B × A
- 每个规格数×非规格数用例都对应一个非规格数×规格数用例
- 确保结果完全一致
```

### 2.2 非规格数乘非规格数测试

#### 2.2.1 FP16 非规格数相乘
```
Group 1: 同号非规格数相乘
- 0x0001 × 0x0001 = 下溢到零
- 0x03FF × 0x03FF = 可能产生规格数
- 0x0200 × 0x0300 = 中等非规格数结果

Group 2: 异号非规格数相乘
- 0x0001 × 0x8001 = 负零或负非规格数
- 0x03FF × 0x83FF = 负数结果

Group 3: 边界组合
- 0x0001 × 0x03FF = 最小×最大非规格数
- 0x03FF × 0x0001 = 验证交换律
```

#### 2.2.2 FP32 非规格数相乘
```
类似FP16的测试模式，但使用32位数值
重点测试渐进下溢的精确性
```

### 2.3 特殊值测试

#### 2.3.1 零值处理
```
Group 1: 规格数 × 零
- 任意正规格数 × +0.0 = +0.0
- 任意正规格数 × -0.0 = -0.0
- 任意负规格数 × +0.0 = -0.0
- 任意负规格数 × -0.0 = +0.0

Group 2: 非规格数 × 零
- 任意正非规格数 × +0.0 = +0.0
- 任意负非规格数 × -0.0 = +0.0

Group 3: 零 × 无穷大 (异常情况)
- +0.0 × +∞ = NaN (无效操作)
- -0.0 × +∞ = NaN (无效操作)
- +0.0 × -∞ = NaN (无效操作)
- -0.0 × -∞ = NaN (无效操作)
```

#### 2.3.2 无穷大处理
```
Group 1: 规格数 × 无穷大
- 任意正规格数 × +∞ = +∞
- 任意正规格数 × -∞ = -∞
- 任意负规格数 × +∞ = -∞
- 任意负规格数 × -∞ = +∞

Group 2: 非规格数 × 无穷大
- 任意正非规格数 × +∞ = +∞
- 任意负非规格数 × -∞ = +∞

Group 3: 无穷大 × 无穷大
- +∞ × +∞ = +∞
- +∞ × -∞ = -∞
- -∞ × -∞ = +∞
```

#### 2.3.3 NaN传播测试
```
Group 1: 输入NaN
- QNaN × 任意数 = QNaN
- SNaN × 任意数 = QNaN (触发无效操作异常)
- 任意数 × QNaN = QNaN
- 任意数 × SNaN = QNaN

Group 2: 运算产生NaN
- 0 × ∞ = QNaN
- ∞ × 0 = QNaN

Group 3: NaN载荷保持
- 验证NaN的载荷信息正确传播
- 测试多个NaN输入时的优先级处理
```

### 2.4 舍入机制测试

#### 2.4.1 舍入到最近偶数
```
Group 1: 精确中点舍入
- 结果正好在两个可表示数中间
- 验证向偶数舍入的正确性

Group 2: 不同舍入位组合
- round_bit = 0, sticky_bit = 0 (精确)
- round_bit = 0, sticky_bit = 1 (向下)
- round_bit = 1, sticky_bit = 0 (中点)
- round_bit = 1, sticky_bit = 1 (向上)

测试用例示例：
- 0x3C00 × 0x0003 (FP16) → 需要舍入的结果
- 验证LSB、round_bit、sticky_bit的正确计算
```

#### 2.4.2 舍入溢出处理
```
Group 1: 舍入导致指数增加
- 尾数舍入后进位到指数位
- 验证指数正确递增

Group 2: 舍入导致溢出到无穷大
- 最大规格数舍入后超出表示范围
- 结果应为相应符号的无穷大
```

### 2.5 异常标志测试

#### 2.5.1 无效操作异常
```
测试用例：
- 0 × ∞ → Invalid flag = 1
- ∞ × 0 → Invalid flag = 1
- SNaN × 任意数 → Invalid flag = 1

验证点：
- 异常标志正确设置
- 结果为QNaN
- 载荷信息处理正确
```

#### 2.5.2 溢出异常
```
测试用例：
- 最大规格数 × 大于1的规格数
- 结果指数超出表示范围

验证点：
- Overflow flag = 1
- Inexact flag = 1 (通常伴随)
- 结果为相应符号的无穷大
```

#### 2.5.3 下溢异常
```
测试用例：
- 小规格数 × 小规格数
- 结果指数小于最小规格化指数

验证点：
- Underflow flag = 1
- 可能设置Inexact flag
- 结果为非规格数或零
```

#### 2.5.4 不精确异常
```
测试用例：
- 任何需要舍入的运算
- 非规格数运算产生规格数

验证点：
- Inexact flag = 1
- 舍入结果正确
```

### 2.6 边界值测试

#### 2.6.1 指数边界
```
FP16边界测试：
- 指数 = 0 (非规格数)
- 指数 = 1 (最小规格数)
- 指数 = 30 (最大规格数)
- 指数 = 31 (无穷大/NaN)

FP32边界测试：
- 指数 = 0 (非规格数)
- 指数 = 1 (最小规格数)
- 指数 = 254 (最大规格数)
- 指数 = 255 (无穷大/NaN)
```

#### 2.6.2 尾数边界
```
非规格数尾数测试：
- 尾数 = 0x001 (最小非零尾数)
- 尾数 = 0x3FF (FP16最大非规格数尾数)
- 尾数 = 0x7FFFFF (FP32最大非规格数尾数)

规格数尾数测试：
- 隐含前导1 + 全0尾数 (精确2的幂)
- 隐含前导1 + 全1尾数 (接近下一个2的幂)
```

### 2.7 压力测试

#### 2.7.1 随机测试
```
测试配置：
- 随机生成规格数和非规格数对
- 所有精度组合
- 运行10,000+测试用例
- 与SoftFloat库对比验证

覆盖策略：
- 约束随机生成确保覆盖所有组合
- 权重调整重点测试关键场景
```

#### 2.7.2 回归测试
```
测试集组成：
- 已知的bug修复用例
- 历史问题重现用例
- 特殊算法验证用例

自动化要求：
- 所有测试用例可重复执行
- 自动对比SoftFloat参考结果
- 异常处理自动验证
```

## 3. 测试环境与工具

### 3.1 仿真环境
```
仿真器：VCS (Synopsys)
语言：SystemVerilog + DPI-C
波形：FSDB格式
覆盖率：代码覆盖率 + 功能覆盖率
```

### 3.2 参考模型
```
参考库：Berkeley SoftFloat-3
接口：DPI-C函数调用
对比精度：位级精确匹配
异常验证：标志位完全一致
```
## 4. 验收标准

### 4.1 功能正确性
- [ ] 所有基本运算测试通过
- [ ] 所有特殊值处理正确
- [ ] 所有异常情况按标准处理
- [ ] 舍入机制完全符合IEEE 754

### 4.2 覆盖率达标
- [ ] 代码覆盖率 ≥ 90%
- [ ] 功能覆盖率 ≥ 95%
- [ ] 特殊值覆盖率 = 100%
- [ ] 边界值覆盖率 = 100%

### 4.3 标准符合性
- [ ] 完全符合IEEE 754-2008标准
- [ ] 与SoftFloat参考库100%一致
- [ ] 异常处理完全正确

**测试计划版本**：V1.0  
**创建日期**：2025年7月1日  
**预计完成**：2025年7月XXX日  
**负责人**：XXX