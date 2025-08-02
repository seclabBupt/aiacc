
##  LDB_ENGINE 设计规范（spec.md）

> **版本**：v1.0  
> **覆盖**：AXI-Lite 主接口、微指令解析、多 SMC 并行访问、字节掩码处理、突发传输支持

---

### 1. 项目概述
实现一个 **微指令驱动的 AXI-Lite 主设备引擎（LDB_ENGINE）**，用于将外部存储器数据搬运到多个 SMC（Scratch Memory Cluster）中的 UR（User Register）中。

支持功能：
- 微指令解析（含 SMC 使能、字节掩码、突发长度）
- AXI-Lite 主接口读取
- 多 SMC 并行写入（基于 SMC ID 和 byte_strb）
- 支持突发长度为 1~65535
- 支持地址间隔可配置（`PARAM_GR_INTLV_ADDR`）

---

### 2. 特性总览

| 特性             | 说明 |
|------------------|------|
| **接口协议**     | AXI-Lite 主接口 |
| **突发长度**     | 最大支持 16-bit `brst` 长度 |
| **SMC 数量**     | 可配置（`PARAM_SMC_CNT`） |
| **UR 宽度**      | 可配置（`PARAM_UR_BYTE_CNT`） |
| **地址间隔**     | 可配置（`PARAM_GR_INTLV_ADDR`） |
| **字节掩码**     | 4-bit `byte_strb`，支持按字节屏蔽 |
| **状态机**       | 4 状态：IDLE → AR_REQ → DATA_RCV → DONE |
| **复位**         | 异步低有效 `rst_n` |

---

### 3. 接口定义（Verilog）

```verilog
module LDB_ENGINE #(
    parameter PARAM_UR_BYTE_CNT   = 16,
    parameter PARAM_GR_INTLV_ADDR = 64,
    parameter PARAM_SMC_CNT       = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // 上行微指令端口
    input  wire                         vld,
    input  wire [5:0]                   smc_strb,
    input  wire [3:0]                   byte_strb,
    input  wire [15:0]                  brst,
    input  wire [31:0]                  gr_base_addr,
    input  wire [$clog2(PARAM_SMC_CNT)-1:0] smc_id,
    input  wire [7:0]                   ur_id,
    input  wire [15:0]                  ur_addr,

    // AXI-Lite 主接口
    output reg  [31:0]                  axi_araddr,
    output reg                          axi_arvalid,
    input  wire                         axi_arready,
    input  wire [31:0]                  axi_rdata,
    input  wire                         axi_rvalid,
    output reg                          axi_rready,

    // 下行微指令端口
    output reg                          done,
    output reg                          vld_down
);
```

---

### 4. 操作行为表

| 信号         | 描述 |
|--------------|------|
| `vld`        | 微指令有效标志（单周期脉冲） |
| `smc_strb`   | 每 bit 控制一个 SMC 是否参与写入 |
| `byte_strb`  | 每 bit 控制 1 Byte 是否写入 UR |
| `brst`       | 突发长度（单位：次） |
| `gr_base_addr` | 外部存储器起始地址 |
| `smc_id`     | 目标 SMC ID（用于突发中的第一个） |
| `ur_id`      | 目标 UR ID（每个 SMC 内 256 个） |
| `ur_addr`    | 目标 UR 地址（16-bit） |

---

### 5. 状态机说明

| 状态       | 行为 |
|------------|------|
| `IDLE`     | 等待 `vld && !active_q` |
| `AR_REQ`   | 发起 AXI-Lite 读地址请求 |
| `DATA_RCV` | 接收数据，按 `byte_strb` 掩码处理后写入 UR |
| `DONE`     | 所有突发完成，拉高 `done` 和 `vld_down` |

---

### 6. 地址计算规则

- 每次突发地址：`addr_q + smc_idx * PARAM_GR_INTLV_ADDR`
- 下一突发地址：`addr_q + PARAM_GR_INTLV_ADDR * PARAM_SMC_CNT`
- 支持跨 SMC 的地址间隔访问

---

### 7. 字节掩码处理

```verilog
for (int j=0; j<4; j++) begin
    if (!byte_strb[j]) 
        masked_data[j*8 +:8] = 8'h0;
end
```

---

### 8. 快速上手

```bash
# 1. 获取文件
ldb.v
tb_ldb.v
run_sim.sh

# 2. 运行仿真
chmod +x run_sim.sh
./run_sim.sh

# 3. 查看结果
sim_output/coverage_report/dashboard.html
```
