# FP16到FP32浮点乘法器

这是一个将两个FP16浮点数相乘并输出FP32结果的硬件设计。

## 文件结构

- `fp16_to_fp32_multiplier.v`: 主设计文件，包含FP16到FP32乘法器的实现
- `fp16_to_fp32_multiplier_tb.v`: 测试平台文件，包含验证用例
- `run_sim.tcl`: ModelSim仿真脚本
- `sim.log`: 仿真日志文件，记录所有仿真输出
- `coverage.log`: 覆盖率日志文件，记录测试覆盖率信息

## 功能特点

- 支持FP16输入（16位浮点数）
- 输出FP32结果（32位浮点数）
- 实现IEEE 754标准的舍入规则
- 支持特殊值处理（如无穷大、零等）
- 流水线设计，2个时钟周期完成计算

## 接口说明

### 输入信号
- `clk`: 时钟信号
- `rst_n`: 低电平有效的复位信号
- `fp16_a`: 第一个FP16输入
- `fp16_b`: 第二个FP16输入
- `valid_in`: 输入有效信号

### 输出信号
- `fp32_out`: FP32输出结果
- `valid_out`: 输出有效信号

## 仿真方法

1. 打开ModelSim
2. 在ModelSim命令行中执行：
```tcl
cd fp16_to_fp32_multiplier
do run_sim.tcl
```

3. 仿真完成后，查看日志文件：
   - `sim.log`: 包含详细的仿真过程和结果
   - `coverage.log`: 包含测试覆盖率信息

## 测试用例

测试平台包含以下测试用例：
1. 正常值测试（如1.0 * 2.0）
2. 负数测试（-1.0 * 2.0）
3. 小数测试（0.5 * 0.5）
4. 边界值测试（最大/最小正数）
5. 舍入测试
6. 特殊值测试（0、无穷大）

## 日志文件说明

### sim.log
- 记录所有仿真过程中的详细信息
- 包含每个测试用例的输入输出
- 记录错误信息和仿真时间
- 格式：时间戳 + 详细信息

### coverage.log
- 记录测试覆盖率信息
- 包含每个测试用例的执行状态
- 记录测试通过/失败情况
- 格式：测试用例ID + 状态 + 详细信息

## 快速入门指南

### 1. 克隆项目
```bash
git clone git@github.com:seclabBupt/aiacc.git
cd /SMC/fp16_to_fp32_multiplier/fp16_to_fp32_multiplier
```

### 2. 运行仿真
```bash
chmod +x run_sim.sh
./run_sim.sh
```

