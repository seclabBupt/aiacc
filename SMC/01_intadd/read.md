# INTADD 整数加法器设计与验证

## 目录
- [项目概述](#项目概述)
- [文件结构](#文件结构)
- [设计逻辑](#设计逻辑)
  - [4+8bit模式 (add8.v)](#14-8bit模式-add8v)
  - [32bit模式 (add32.v)](#232bit模式-add32v)
  - [顶层模块 (intadd.v)](#3-顶层模块-intaddv)
- [测试平台 (intadd_tb.v)](#测试平台-intadd_tbv)
  - [测试逻辑](#测试逻辑)
  - [详细计算过程记录](#详细计算过程记录)
- [使用方法](#使用方法)
- [工程应用示例](#工程应用示例)
- [测试结果](#测试结果)
- [设计特点](#设计特点)

---

## 项目概述
本项目实现了一个支持多种精度（4+8bit模式和32bit模式）的整数加法器（INTADD），包含完整的RTL设计、C语言参考模型和验证环境。项目使用SystemVerilog实现硬件设计，通过DPI-C接口将C参考模型集成到测试平台中进行验证。

## 文件结构
<details>
<summary>点击展开文件结构</summary>

| 文件 | 说明 |
|------|------|
| `add8.v` | 4+8bit模式加法器模块 |
| `add32.v` | 32bit模式加法器模块 |
| `intadd.v` | 顶层加法器模块 |
| `intadd_interface.c` | C语言参考模型 |
| `intadd_tb.v` | 测试平台 |
| `run_simu.sh` | 仿真运行脚本 |

</details>

## 设计逻辑

### 1. 4+8bit模式 (add8.v)
<details>
<summary>点击展开详细逻辑</summary>

支持32个独立的8位加法器  

**输入处理**：
- 每个源寄存器分为32个4位段
- 根据符号控制信号进行符号/零扩展  

**运算过程**：
- 三个8位数相加（扩展后的源操作数）
- 结果饱和处理（-128~127）  

**输出处理**：
- 结果拆分为两个4位段（低4位存入dst0，高4位存入dst1）
- 状态寄存器：固定输出0（不更新）  

</details>

### 2. 32bit模式 (add32.v)
<details>
<summary>点击展开详细逻辑</summary>

支持4个独立的32位加法器  

**输入处理**：
- 每个源寄存器分为4个32位段
- 根据符号控制信号进行符号/零扩展  

**运算过程**：
- 两个64位数相加（扩展后的源操作数）
- 溢出时饱和处理（0x7FFFFFFF或0x80000000）  

**输出处理**：
- 32位结果存入dst  

**状态寄存器**：
- 每组生成3位状态（gt, eq, ls）
- 共占用12位（4组×3位）  

</details>

### 3. 顶层模块 (intadd.v)
<details>
<summary>点击展开详细逻辑</summary>

**控制信号解码**：
- `cru_intadd[10]`: 指令有效
- `cru_intadd[9:8]`: src0精度
- `cru_intadd[7:6]`: src1精度
- `cru_intadd[5:4]`: src2精度
- `cru_intadd[3:1]`: 符号控制
- `cru_intadd[0]`: 状态更新使能  

**模式选择**：
- 4+8bit模式：当所有精度为00时
- 32bit模式：当src0和src1精度为11时  

**输出选择**：
- 根据当前模式选择输出
- 非目标模式输出0  

</details>

## 测试平台 (intadd_tb.v)

### 测试逻辑
<details>
<summary>点击展开详细测试逻辑</summary>

**初始化**：
- 复位系统，初始化信号  

**定向测试**：
- 32bit正溢出测试：0x7FFFFFFF + 1
- 32bit负溢出测试：0x80000000 + (-1)  

**随机测试**：
- 1000次4+8bit模式随机测试  

**结果比较**：
- 调用DPI-C参考模型
- 详细记录每个lane的计算过程
- 比较RTL和参考模型输出  

**覆盖率收集**：
- 行覆盖、条件覆盖、状态机覆盖  

</details>

### 详细计算过程记录
<details>
<summary>点击展开</summary>

**32bit模式**：
- 无符号和有符号求和
- 进位和溢出情况
- RTL与参考模型对比  

**4+8bit模式**：
- 4bit段和8bit段计算
- 饱和处理细节
- 多输出结果对比  

</details>

## 使用方法
<details>
<summary>点击展开操作步骤</summary>

1. **准备环境**：
   - VCS编译器
   - gcc
   - Linux环境  

2. **运行仿真**：
```bash
chmod +x run_simu.sh
./run_simu.sh
查看结果：

bash
复制
编辑
sim_output/
├── result.txt         # 详细测试结果
├── coverage.txt       # 覆盖率文本报告
└── coverage_html/     # 覆盖率HTML报告
提高覆盖率：

verilog
复制
编辑
for (i = 0; i < 1000; i++) begin
    // 随机测试代码
end
</details>
工程应用示例
<details> <summary>点击展开示例代码</summary>
4+8bit模式
python
复制
编辑
dst0, dst1 = intadd(src0, src1, src2, cru_intadd)
for i in range(32):
    result = (dst1[i] << 4) | dst0[i]
32bit模式
python
复制
编辑
dst, st = intadd(src0, src1, 0, cru_intadd)
for i in range(4):
    gt = st[i*32 + 2]
    eq = st[i*32 + 1]
    ls = st[i*32 + 0]
</details>
测试结果
<details> <summary>点击展开测试数据</summary>
测试类型	通过情况
定向测试	100%
随机测试 (1000次)	100%
覆盖率	行覆盖、条件覆盖、状态机覆盖均接近100%

</details>
设计特点
灵活的多精度支持（4+8bit和32bit）

饱和处理防止溢出

提供详细状态反馈

高覆盖率验证

可扩展架构

markdown
复制
编辑
