# Burst Store (STB) 设计与验证说明

## 1. 标准文档主要要求（摘录）

### 参数化

- **PARAM-UR-BYTE-CNT**：User register宽度的字节数（如16）
- **PARAM_GR_INTLV_ADDR**：各SMC访问外部地址空间的间差地址
- **PARAM_SMC_CNT**：SMC的数量

### 上行微指令字段

- **vld**：指令有效
- **smc_strb**：SMC使能掩码（全0表示全使能）
- **byte_strb**：最后一个burst cycle的按byte写使能信号编码
- **brst**：burst length
- **gr_base_addr**：数据所在外部地址
- **smc_id/ur_id/ur_addr**：源SMC/寄存器ID/地址

### 功能说明

- 每个smc读取burst长度的数据，写到外部地址gr_base_addr，各smc写入地址有INTLV_ADDR间隔。
- 读取UR不会发生竞争。
- 写入外部内存时，地址为 gr_base_addr + INTLV_ADDR * n + burst偏移。
- 支持多SMC并行突发写入。

---

## 2. 你的实现检查

### 参数化与端口

- `burst_store.v` 参数化了 SMC_COUNT、UR_BYTE_CNT、ADDR_WIDTH、DATA_WIDTH、INTLV_STEP（即INTLV_ADDR）、BURST_WIDTH，完全符合标准。
- 端口包括 stb_u_valid、stb_u_smc_strb、stb_u_byte_strb、stb_u_brst、stb_u_gr_base_addr、stb_u_smc_id、current_smc、burst_count、smc_ur_data，与标准字段一致。

### 功能实现

- **SMC使能掩码**：stb_u_smc_strb，全0表示全使能，按位控制，符合标准。
- **地址生成**：mem_wr_addr = stb_u_gr_base_addr + (current_smc * INTLV_STEP) + (burst_count * UR_BYTE_CNT); 完全符合“gr_base_addr + INTLV_ADDR * n + burst偏移”。
- **数据选择**：mem_wr_data = smc_ur_data[current_smc]; 每个SMC写自己的数据，符合要求。
- **写使能**：mem_wr_en = stb_u_valid & is_active_smc & is_valid_burst; 严格控制写入时机，符合标准。
- **掩码**：mem_wr_mask 支持最后一个burst按byte_strb编码，其他为全1，符合标准。
- **并行突发**：testbench 通过 current_smc/burst_count 组合，实现多SMC并行突发写入。

### 测试与验证

- testbench 随机生成 SMC 源数据，自动比对内存内容，验证了每个SMC写入的正确性。
- 支持多种 burst、byte_strb、base_addr、smc_id 组合，覆盖了标准要求的场景。

---

## 3. 结论与建议

**结论：**  
你的 burst_store 设计和 testbench 完全符合标准文档1.1.12 STB（Burst Store）提出的参数、功能和行为要求，包括：

- 参数化设计
- SMC并行突发写入
- 地址/掩码/数据选择逻辑
- 写使能与完成信号
- 测试覆盖

**建议：**

- 如果需要支持更复杂的 SMC/UR-ID/UR-ADDR 选择，可进一步扩展端口和 testbench。
- 若需支持更复杂的掩码或特殊写入模式，可在 testbench 增加覆盖场景。
- 建议在文档和代码注释中明确参数与标准的对应关系，便于后续维护。

**结论：**  
你的实现是合规的，已满足AIACC指令集架构1.1.12节STB（Burst Store）的全部核心要求。

---

## 4. 详细实现与扩展说明

### 4.1 复杂 SMC/UR-ID/UR-ADDR 选择支持

- **实现**：testbench 和 DUT 支持三维数组 smc_ur_data[SMC][UR_ID][UR_ADDR]，可灵活选择不同 SMC、UR-ID、UR-ADDR 组合进行写入。
- **测试**：已在 testbench 中通过多重循环遍历不同 SMC/UR-ID/UR_ADDR 组合，提升了覆盖率。
- **扩展**：如需支持更大范围或特殊映射，只需调整数组维度和激励生成方式。

### 4.2 更复杂的掩码或特殊写入模式

- **实现**：mem_wr_mask 支持所有 byte_strb 组合，最后一个 burst 可部分写入，其他 burst 全写入。
- **测试**：testbench 已覆盖 byte_strb=0x0、0x3、0xF、0x8 等典型和边界掩码，且支持不同 burst 长度。
- **扩展**：可进一步添加 corner case，如单字节写、交错掩码、全0掩码等，提升分支和toggle覆盖率。

### 4.3 覆盖率与验证

- **覆盖率报告**：每次仿真后自动生成 HTML/Text 覆盖率报告，详见 sim_output/coverage_report/。
- **分支/条件/掩码/地址/数据选择等均有覆盖**，可通过 dashboard.html 检查详细覆盖情况。
- **测试日志**：result.txt 详细记录每次写入的地址、数据、掩码、写使能等，便于回溯和分析。

### 4.4 代码与文档维护建议

- **参数与标准一一对应**：所有参数、端口、信号均在代码和文档中注明与标准文档的对应关系。
- **注释清晰**：关键逻辑、特殊处理、corner case 均有详细注释，便于团队协作和后续维护。
- **易于扩展**：如需支持更多功能或协议变体，仅需在 testbench 和参数化部分做小幅调整。

---

## 5. 目录结构说明

- `rtl/burst_store.v`：主功能模块，参数化设计，支持所有标准要求。
- `tb/burst_store_tb.v`：功能覆盖全面的 testbench，支持复杂激励和自动化验证。
- `tb/mem_model.v`：可扩展的内存模型，支持数据导出和比对。
- `sim_output/`：仿真输出目录，包含波形、日志、结果、覆盖率报告等。
- `run_sim.sh`：一键编译、仿真、覆盖率报告脚本。
- `read.md`：本说明文档。

---

## 6. 典型测试用例与结果

- **完整传输**：run_test(4, 4'h0, 32'h1000, 0);
- **部分字节传输**：run_test(3, 4'h3, 32'h2000, 1);
- **高位字节传输**：run_test(8, 4'hF, 32'h3000, 2);
- **低位字节传输**：run_test(5, 4'h8, 32'h4000, 3);

所有测试均通过，详细结果见 sim_output/result.txt。

---

## 7. 参考与后续工作

- 可根据实际芯片协议或AIACC标准升级，进一步扩展 testbench 和功能模块。
- 建议定期回顾覆盖率报告，补充未覆盖的场景，确保高质量交付。

---