# RUR 模块设计规范（RUR_spec.md）

> **版本**：v1.1  
> **作者**：cypher  
> **覆盖**：128-bit 多通道数据选择器，支持 8×16-byte 子通道，按地址+掩码提取数据  

---

## 1. 项目概述
实现 **128-bit 多通道数据选择器（RUR）**，根据 97-bit 指令格式，从 8×256×128-bit 的 UR-RAM 中提取指定字节并掩码写入结果寄存器，支持按通道独立寻址与掩码控制。

---

## 2. 特性总览
| 特性           | 说明 |
|----------------|------|
| **通道数**     | 8 个独立通道（bank），每通道 256 条 128-bit 记录 |
| **指令宽度**   | 97-bit |
| **输出宽度**   | 128-bit |
| **寻址粒度**   | 字节（8-bit） |
| **掩码粒度**   | 16 组 8-bit 掩码（每组 4-bit 子地址 + 1-bit 有效位） |
| **延迟**       | 单周期（非流水线） |
| **复位**       | 异步低有效 `rst_n` |

---

## 3. 接口定义
```verilog
module RUR #(
    parameter LOCAL_SMC_ID = 5'd0
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [96:0] cru_rur,   // 97-bit 上行指令
    output reg  [127:0] dr_rur_d  // 128-bit 输出
);
```

---

## 4. 指令格式（97-bit）

| 位段        | 名称    | 描述                                  |
|-------------|---------|---------------------------------------|
| [96]        | vld     | 指令有效                              |
| [95:91]     | smc_id  | 目标 SMC ID（需匹配 LOCAL_SMC_ID）    |
| [90:88]     | ur_id   | 通道号（0–7）                         |
| [87:80]     | ur_addr | 通道内地址（0–255）                   |
| [79:0]      | lo      | 16 组 {4-bit 子地址, 1-bit 有效}     |

---

## 5. 功能描述
- 当 `vld=1` 且 `smc_id==LOCAL_SMC_ID`：
  1. 从 `ur_ram[ur_id][ur_addr]` 读取 128-bit 数据。
  2. 遍历 16 组 `lo[i]`：
     - 若 `lo[i].vld=1`，则将 `ram[8*lo[i].addr +: 8]` 写入 `dr_rur_d[8*i +: 8]`。
  3. 结果采用 OR 累积方式更新（非覆盖）。

---

## 6. UR-RAM 结构

```verilog
( ram_style = "block" )
reg [127:0] ur_ram [0:7][0:255]; // 8×256×128-bit
```
---

## 7. 复位行为
- `rst_n=0` 时：
  - `dr_rur_d` 清零。
  - UR-RAM 内容保持不变（由测试平台初始化）。

---

## 8. 验证与交付

| 项           | 内容                                      |
|--------------|-------------------------------------------|
| 黄金模型     | SystemVerilog 参考模型（tb_rur.v）        |
| 仿真脚本     | `run_sim.sh`（VCS + Verdi + URG）         |
| 覆盖率       | 行≥95%，分支≥90%，功能点 100%             |
| 测试用例     | 随机/定向指令、边界地址、全掩码/零掩码    |
| 日志         | `sim.log`（pass/fail 统计）               |
| 波形         | `tb.vpd.fsdb`                             |

---

## 9. 快速上手
```bash
# 1. 获取所有文件
rur.v
tb_rur.v
run_sim.sh

# 2. 一键运行
chmod +x run_sim.sh
./run_sim.sh
# 结果查看 sim_output/coverage_report/dashboard.html
```
