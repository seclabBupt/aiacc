# INTADD 整数加法器技术规格文档

## 1. 概述

### 1.1 模块功能
INTADD 模块是一个支持子字并行的整数加法器模块，支持 4-bit + 8-bit 和 32-bit + 32-bit；模块具有精度选择信号和符号位信号；对于溢出情况置F处理。

### 1.2 设计目标
- 支持多种精度模式下的无符号或有符号整数加法
- 支持子字并行
- 具备溢出检测与结果裁剪（饱和）能力
- 支持4bit + 4bit
- 支持8bit + 8bit

## 2. 接口规格

### 2.1 顶层模块接口

```verilog
module intadd (
    input  wire [127:0] src_reg0,      // 源操作数0（支持4/32bit对齐）
    input  wire [127:0] src_reg1,      // 源操作数1（支持4/32bit对齐）
    input  wire [127:0] src_reg2,      // 源操作数2（仅用于add8）
    input  wire [1:0]   precision_s0,  // 源操作数0精度
    input  wire [1:0]   precision_s1,  // 源操作数1精度
    input  wire [1:0]   precision_s2,  // 源操作数2精度
    input  wire         sign_s0,       // 源操作数0是否为有符号
    input  wire         sign_s1,       // 源操作数1是否为有符号
    input  wire         sign_s2,       // 源操作数2是否为有符号
    input  wire         inst_valid,    // 指令有效信号
    output wire [127:0] dst_reg0,      // 运算结果低位或最终结果
    output wire [127:0] dst_reg1       // 运算结果高位（仅用于add8）
);
```

### 2.2 信号定义

| 信号名 | 位宽 | 方向 | 功能描述 |
|-----|---|------|------|
| src_reg0 | 128 | Input | 第一个源操作数 |
| src_reg1 | 128 | Input | 第二个源操作数 |
| src_reg2 | 128 | Input | 第三个源操作数（仅 add8 模式使用） |
| precision_s0 | 2 | Input | 操作数0的精度（2'b00=4bit，2'b11=32bit） |
| precision_s1 | 2 | Input | 操作数1的精度 |
| precision_s2 | 2 | Input | 操作数2的精度 |
| sign_s0 | 1 | Input | 操作数0是否为有符号 |
| sign_s1 | 1 | Input | 操作数1是否为有符号 |
| sign_s2 | 1 | Input | 操作数2是否为有符号（仅 add8） |
| inst_valid | 1 | Input | 当前指令是否有效 |
| dst_reg0 | 128 | Output | 运算结果低位或最终结果 |
| dst_reg1 | 128 | Output | 运算结果高位（仅 add8 输出高位） |

## 3. 功能规格

### 3.1 add8 模块（4-bit + 8-bit 加法器）
- 每组 4-bit 的 src0 与拼接后的 8-bit {src2,src1} 相加
- 支持有符号和无符号输入，按 sign_sX 控制
- 运算结果为 8-bit，分为高 4 位和低 4 位输出
- 溢出时饱和至 8'hFF

### 3.2 add32 模块（32-bit 加法器）
- src0 和 src1 各拆为 4 个 32-bit 块，分别加法
- 支持有符号与无符号，符号扩展由 sign_sX 控制
- 溢出时饱和至 32'hFFFFFFFF

### 3.3 顶层模块选择逻辑

| 条件 | 启用模块 | 输出 |
|----|------|----|
| precision_s0 == precision_s1 == precision_s2 == 2'b00 | add8  | dst_reg0, dst_reg1 |
| precision_s0 == precision_s1 == 2'b11 | add32  | dst_reg0 |
| 其他组合 | 无效  | 全零 |

## 4. 异常与边界处理

| 情况 | 处理方式 |
|----|----|
| 正溢出（>最大值） | 输出饱和值（全1） |
| 负溢出（<最小值） | 输出饱和值（全1） |
| 输入组合不合法 | 输出 0 |

## 5. 时序与性能

### 5.1 时序说明
- 组合逻辑实现，所有计算在单周期内完成
- 建议时钟周期 > add32 中 33-bit 加法延迟

### 5.2 并行能力
- add8: 32组独立4-bit单元并行执行
- add32: 4组独立32-bit加法单元并行执行

## 6. 验证说明

### 6.1 验证用例覆盖
- 初始化测试
- 基本功能测试（4-bit+8-bit 与 32-bit+32-bit 加法）
- 4-bit+8-bit情况下，符号与数据完全随机测试
- 32-bit+32-bit情况下，符号与数据完全随机测试
- 完全随机测试（精度选择、符号、数据全都随机，100轮）
- 边界情况测试（最大正数、最小负数）

### 6.2 验证文档
- 测试计划和用例
- 覆盖率报告
- 仿真日志和波形
- 验证总结报告

---

**文档版本**：V1.1  
**创建日期**：2025年5月13日  
**最后更新**：2025年7月20日  