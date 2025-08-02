`timescale 1ns/1ps

module LDB_ENGINE #(
    parameter PARAM_UR_BYTE_CNT  = 16,            // 用户寄存器宽度（Byte）
    parameter PARAM_GR_INTLV_ADDR= 64,            // 地址间隔
    parameter PARAM_SMC_CNT      = 4              // SMC 数量
)(
    input  wire                         clk,
    input  wire                         rst_n,        // 低有效

    // 上行微指令端口
    input  wire                         vld,          // 本拍指令有效
    input  wire [5:0]                   smc_strb,     // SMC使能位
    input  wire [3:0]                   byte_strb,    // Byte掩码编码
    input  wire [15:0]                  brst,         // burst length
    input  wire [31:0]                  gr_base_addr, // 外部起始地址
    input  wire [$clog2(PARAM_SMC_CNT)-1:0] smc_id,   // 目的SMC ID
    input  wire [7:0]                   ur_id,        // 目的UR ID
    input  wire [15:0]                  ur_addr,      // 目的UR地址

    // AXI-Lite接口
    output reg  [31:0]                  axi_araddr,
    output reg                          axi_arvalid,
    input  wire                         axi_arready,
    input  wire [31:0]                  axi_rdata,
    input  wire                         axi_rvalid,
    output reg                          axi_rready,

    // 下行微指令端口
    output reg                          done,         // 指令完成
    output reg                          vld_down      // 下行有效
);

    // 内部常量
    localparam UR_DATA_WIDTH = PARAM_UR_BYTE_CNT * 8;
    localparam MAX_BURST     = 16'hFFFF;

    // 寄存器组
    reg [15:0] burst_cnt_q, burst_cnt_d;
    reg [31:0] addr_q, addr_d;
    reg [5:0]  smc_strb_q;
    reg [3:0]  byte_strb_q;
    reg [$clog2(PARAM_SMC_CNT)-1:0] smc_id_q;
    reg [7:0]  ur_id_q;
    reg [15:0] ur_addr_q;
    reg        active_q, active_d;
    reg [2:0]  smc_idx_q, smc_idx_d;  // 新增：SMC遍历索引

    reg [UR_DATA_WIDTH-1:0] ur_wdata [0:PARAM_SMC_CNT-1];
    reg [PARAM_SMC_CNT-1:0] ur_we;

    // 状态机重构（4状态）
    typedef enum logic [2:0] {
        IDLE      = 3'b000,
        AR_REQ    = 3'b001,  // 地址请求
        DATA_RCV  = 3'b010,  // 数据接收
        DONE      = 3'b100   // 传输完成
    } state_t;

    state_t state_q, state_d;

    // 组合逻辑：状态转移与控制
    always @(*) begin
        // 默认值
        state_d       = state_q;
        active_d      = active_q;
        burst_cnt_d   = burst_cnt_q;
        addr_d        = addr_q;
        smc_idx_d     = smc_idx_q;
        
        axi_araddr    = 32'h0;
        axi_arvalid   = 1'b0;
        axi_rready    = 1'b0;
        done          = 1'b0;
        vld_down      = 1'b0;
        ur_we         = '0;

        case (state_q)
            IDLE: begin
                if (vld && !active_q) begin
                    active_d      = 1'b1;
                    burst_cnt_d   = brst;
                    addr_d        = gr_base_addr;
                    smc_idx_d     = 0;
                    state_d       = AR_REQ;
                end
            end

            AR_REQ: begin
                axi_arvalid = 1'b1;
                // 动态计算地址：基址 + SMC索引 * 间隔
                axi_araddr  = addr_q + (PARAM_GR_INTLV_ADDR * smc_idx_q); 
                
                if (axi_arvalid && axi_arready) begin
                    state_d = DATA_RCV;  // 地址握手成功
                end
            end

            DATA_RCV: begin
                axi_rready = 1'b1;
                
                if (axi_rvalid && axi_rready) begin
                    // --- 字节掩码逻辑 ---
                    reg [31:0] masked_data = axi_rdata;
                    for (int j=0; j<4; j++) begin
                        if (!byte_strb_q[j]) 
                            masked_data[j*8 +:8] = 8'h0; // 屏蔽无效字节
                    end
                    // 仅写入使能的SMC
                    if (smc_strb_q[smc_idx_q]) begin
                        ur_wdata[smc_idx_q] = {>>{masked_data}}; // 小端转换
                        ur_we[smc_idx_q]    = 1'b1;
                    end

                    // 更新SMC索引
                    if (smc_idx_q == PARAM_SMC_CNT-1) begin
                        smc_idx_d = 0;
                        burst_cnt_d = burst_cnt_q - 1;
                        addr_d = addr_q + PARAM_GR_INTLV_ADDR * PARAM_SMC_CNT;
                    end else begin
                        smc_idx_d = smc_idx_q + 1;
                    end

                    // 突发结束检测
                    if (burst_cnt_d == 0 && smc_idx_d == 0) begin
                        state_d = DONE;
                    end else begin
                        state_d = AR_REQ;  // 继续下一传输
                    end
                end
            end

            DONE: begin
                done     = 1'b1;
                vld_down = 1'b1;
                if (!axi_rvalid) begin      // 确保无残留数据
                    active_d = 1'b0;
                    state_d  = IDLE;
                end
            end

            default: state_d = IDLE;
        endcase
    end

    // 时序逻辑：寄存器更新
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q      <= IDLE;
            active_q     <= 1'b0;
            burst_cnt_q  <= '0;
            addr_q       <= '0;
            smc_idx_q    <= '0;
            smc_strb_q   <= '0;
            byte_strb_q  <= '0;
            smc_id_q     <= '0;
            ur_id_q      <= '0;
            ur_addr_q    <= '0;
        end else begin
            state_q      <= state_d;
            active_q     <= active_d;
            burst_cnt_q  <= burst_cnt_d;
            addr_q       <= addr_d;
            smc_idx_q    <= smc_idx_d;
            if (vld && !active_q) begin
                smc_strb_q  <= smc_strb;
                byte_strb_q <= byte_strb;
                smc_id_q    <= smc_id;
                ur_id_q     <= ur_id;
                ur_addr_q   <= ur_addr;
            end
        end
    end

    // UR寄存器实例化
    genvar g;
    generate
        for (g = 0; g < PARAM_SMC_CNT; g = g + 1) begin : gen_ur
            reg [UR_DATA_WIDTH-1:0] ur [0:255];
            always @(posedge clk) begin
                if (ur_we[g])
                    ur[ur_addr_q] <= ur_wdata[g];
            end
        end
    endgenerate

endmodule