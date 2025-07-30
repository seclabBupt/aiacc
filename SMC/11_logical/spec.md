# Logical/Shift Unit 设计规范 (logical_spec.md)

> **版本**：v1.0  
> **覆盖**：32-bit/16-bit 逻辑、移位、选择操作，单周期执行  

---

## 1. 项目概述
实现 **32-bit 多功能逻辑/移位单元**，支持：
- **逻辑操作**：AND / OR / XOR / NOT / COPY  
- **选择操作**：GT / EQ / LS（基于浮点比较结果）  
- **移位操作**：逻辑/算术/循环移位，可配置方向（左/右）与精度（32/16-bit）  
- **接口**：32-bit 输入×2，32-bit 输出，单周期延迟

---

## 2. 特性总览
| 特性           | 说明 |
|----------------|------|
| **精度模式**   | 1 位 `logical_precision_i`：0=16-bit，1=32-bit |
| **操作码**     | 4 位 `logical_op_i`，支持 11 种操作 |
| **移位方向**   | 1 位 `shift_dir_i`：0=左移，1=右移 |
| **延迟**       | 单周期（非流水线） |
| **比较来源**   | 3 位 `fpadd_status_i`，来自浮点单元比较结果 |
| **复位**       | 异步低有效 `rst_n` |

---

## 3. 接口定义
```verilog
module logical_unit (
    input wire clk,
    input wire rst_n,
    input wire logical_vld_i,          // 操作有效
    input wire [3:0] logical_op_i,     // 操作码
    input wire logical_precision_i,    // 精度选择
    input wire shift_dir_i,            // 移位方向
    input wire [31:0] logical_src0_i,  // 源操作数 A
    input wire [31:0] logical_src1_i,  // 源操作数 B / 移位量
    input wire [2:0] fpadd_status_i,   // 浮点比较结果 {GT,EQ,LS}
    output reg logical_done_o,         // 完成标志
    output reg [31:0] logical_dst_o    // 结果
);
```

---

## 4. 操作码表
| 操作码 | 名称           | 描述                             |
|--------|----------------|----------------------------------|
| 0000   | OP_AND         | 按位与                           |
| 0001   | OP_OR          | 按位或                           |
| 0010   | OP_XOR         | 按位异或                         |
| 0011   | OP_NOT         | 按位取反                         |
| 0100   | OP_COPY        | 复制 src0                        |
| 0101   | OP_SELECT_GT   | 若 GT=1 选 src0，否则选 src1     |
| 0110   | OP_SELECT_EQ   | 若 EQ=1 选 src0，否则选 src1     |
| 0111   | OP_SELECT_LS   | 若 LS=1 选 src0，否则选 src1     |
| 1000   | OP_LOGIC_SHIFT | 逻辑移位（无符号）               |
| 1001   | OP_ARITH_SHIFT | 算术移位（有符号，含符号扩展）   |
| 1010   | OP_ROT_SHIFT   | 循环移位（旋转）                 |

---

## 5. 移位操作细节
### 5.1 逻辑移位 (OP_LOGIC_SHIFT)
- **32-bit**  
  - 左移：`src0 << (src1 & 31)`  
  - 右移：`src0 >> (src1 & 31)`  
- **16-bit**  
  - 左移：`{16'b0, src0[15:0] << (src1 & 15)}`  
  - 右移：`{16'b0, src0[15:0] >> (src1 & 15)}`

### 5.2 算术移位 (OP_ARITH_SHIFT)
- **32-bit**  
  - 左移：`$signed(src0) <<< (src1 & 31)`  
  - 右移：`$signed(src0) >>> (src1 & 31)`  
- **16-bit**  
  - 左移：`{16'b0, $signed(src0[15:0]) <<< (src1 & 15)}`  
  - 右移：结果符号扩展至 32-bit（高位补符号位）

### 5.3 循环移位 (OP_ROT_SHIFT)
- **32-bit**  
  - `rot = (src0 >> n) | (src0 << (32-n))`，其中 `n = src1 & 31`  
- **16-bit**  
  - `rot = (src0[15:0] >> n) | (src0[15:0] << (16-n))`，结果右对齐至低 16 位，高位补 0

---

## 6. DPI-C 黄金模型
| 功能                | 接口 |
|---------------------|------|
| **舍入模式设置**    | `set_softfloat_rounding_mode(0)` |
| **清除异常标志**    | `clear_softfloat_flags()` |
| **获取异常标志**    | `get_softfloat_flags()` |
| **32-bit 比较**     | `fp32_compare_softfloat(a, b)` → {0=EQ,1=LS,2=GT} |
| **16-bit 比较**     | `fp16_compare_softfloat(a, b)` → {0=EQ,1=LS,2=GT} |

---

## 7. 验证与交付
| 项            | 内容 |
|---------------|------|
| **黄金模型**  | SoftFloat-DPI (`softfloat_dpi.c`) |
| **仿真脚本**  | `run_sim.sh`（VCS + Verdi + Urg） |
| **覆盖率**    | 行≥95%，分支≥90%，功能点 100% |
| **测试用例**  | 逻辑、移位、选择、边界、符号扩展、NaN/Inf |
| **日志**      | `sim.log`（pass/fail 统计） |
| **波形**      | `tb_logical_unit.vcd` |

---

## 8. 快速上手
```bash
# 1. 获取所有文件
logical.v
tb_logical.v
softfloat_dpi.c
run_sim.sh

# 2. 一键运行
chmod +x run_sim.sh
./run_sim.sh
# 结果查看 sim_output/coverage_report/dashboard.html
```

---
```