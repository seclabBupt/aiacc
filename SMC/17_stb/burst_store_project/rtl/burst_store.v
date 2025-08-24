`timescale 1ns/1ps

module burst_store #(
    parameter SMC_COUNT      = 6,        // SMC 数量
    parameter UR_BYTE_CNT    = 16,       // UR 寄存器字节宽度
    parameter ADDR_WIDTH     = 32,       // 地址宽度
    parameter DATA_WIDTH     = 128,      // 数据宽度
    parameter INTLV_STEP     = 64,       // 地址交错步长
    parameter BURST_WIDTH    = 8,        // Burst长度位宽
    parameter UR_ADDR_WIDTH  = 11,       // UR 地址宽度
    parameter UR_ID_WIDTH    = 3         // UR ID 宽度
)(
    // 时钟和复位
    input  wire                         clk,
    input  wire                         rst_n,
    
    // 上行微指令接口
    input  wire                         stb_u_valid,           // 指令有效
    input  wire [SMC_COUNT-1:0]         stb_u_smc_strb,        // SMC 使能掩码
    input  wire [3:0]                   stb_u_byte_strb,       // 字节掩码编码
    input  wire [BURST_WIDTH-1:0]       stb_u_brst,            // Burst 长度
    input  wire [ADDR_WIDTH-1:0]        stb_u_gr_base_addr,    // 基地址
    input  wire [UR_ID_WIDTH-1:0]       stb_u_ur_id,           // UR ID
    input  wire [UR_ADDR_WIDTH-1:0]     stb_u_ur_addr,         // UR 起始地址
    
    // SMC 寄存器接口
    output reg                          ur_re,                 // UR 读使能
    output reg  [UR_ID_WIDTH-1:0]       ur_id,                 // UR ID
    output reg  [UR_ADDR_WIDTH-1:0]     ur_addr,               // UR 读地址
    input  wire [DATA_WIDTH-1:0]        ur_rdata,              // UR 读数据
    
    // AXI 主接口
    output reg                          axi_awvalid,           // AXI 写地址有效
    output wire [ADDR_WIDTH-1:0]        axi_awaddr,            // AXI 写地址
    input  wire                         axi_awready,           // AXI 写地址就绪
    
    output reg                          axi_wvalid,            // AXI 写数据有效
    output wire [DATA_WIDTH-1:0]        axi_wdata,             // AXI 写数据
    output wire [UR_BYTE_CNT-1:0]       axi_wstrb,             // AXI 写字节使能
    output reg                          axi_wlast,             // AXI 写最后数据标志
    input  wire                         axi_wready,            // AXI 写数据就绪
    
    input  wire                         axi_bvalid,            // AXI 写响应有效
    output reg                          axi_bready,            // AXI 写响应就绪
    
    // 下行响应接口
    output reg                          stb_d_valid,           // 响应有效
    output reg                          stb_d_done             // 操作完成
);

    // 状态定义
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        INIT        = 3'b001,
        AWVALID     = 3'b010,
        WVALID      = 3'b011,
        WAIT_RESP   = 3'b100,
        DONE        = 3'b101
    } state_t;
    
    // 内部寄存器
    state_t current_state, next_state;
    reg [$clog2(SMC_COUNT)-1:0] current_smc;
    reg [BURST_WIDTH-1:0] burst_count;
    reg [SMC_COUNT-1:0] smc_strb_reg;
    reg [3:0] byte_strb_reg;
    reg [BURST_WIDTH-1:0] brst_reg;
    reg [ADDR_WIDTH-1:0] gr_base_addr_reg;
    reg [UR_ID_WIDTH-1:0] ur_id_reg;
    reg [UR_ADDR_WIDTH-1:0] ur_addr_reg;
    
    // 字节掩码解码器
    wire [UR_BYTE_CNT-1:0] byte_mask;
    assign byte_mask = 
        (byte_strb_reg == 4'h0) ? 16'b1111_1111_1111_1111 :
        (byte_strb_reg == 4'h1) ? 16'b0000_0000_0000_0001 :
        (byte_strb_reg == 4'h2) ? 16'b0000_0000_0000_0011 :
        (byte_strb_reg == 4'h3) ? 16'b0000_0000_0000_0111 :
        (byte_strb_reg == 4'h4) ? 16'b0000_0000_0000_1111 :
        (byte_strb_reg == 4'h5) ? 16'b0000_0000_0001_1111 :
        (byte_strb_reg == 4'h6) ? 16'b0000_0000_0011_1111 :
        (byte_strb_reg == 4'h7) ? 16'b0000_0000_0111_1111 :
        (byte_strb_reg == 4'h8) ? 16'b0000_0000_1111_1111 :
        (byte_strb_reg == 4'h9) ? 16'b0000_0001_1111_1111 :
        (byte_strb_reg == 4'hA) ? 16'b0000_0011_1111_1111 :
        (byte_strb_reg == 4'hB) ? 16'b0000_0111_1111_1111 :
        (byte_strb_reg == 4'hC) ? 16'b0000_1111_1111_1111 :
        (byte_strb_reg == 4'hD) ? 16'b0001_1111_1111_1111 :
        (byte_strb_reg == 4'hE) ? 16'b0011_1111_1111_1111 :
        (byte_strb_reg == 4'hF) ? 16'b0111_1111_1111_1111 :
        16'b1111_1111_1111_1111; // 默认值
    
    // 地址计算
    wire [ADDR_WIDTH-1:0] base_offset = gr_base_addr_reg + (current_smc * INTLV_STEP);
    wire [ADDR_WIDTH-1:0] burst_offset = burst_count * UR_BYTE_CNT;
    assign axi_awaddr = base_offset + burst_offset;
    
    // 数据直接来自 UR
    assign axi_wdata = ur_rdata;
    
    // 字节掩码选择
    assign axi_wstrb = (burst_count == brst_reg - 1) ? byte_mask : {UR_BYTE_CNT{1'b1}};
    
    // SMC 使能检查
    wire all_smc_enable = (smc_strb_reg == {SMC_COUNT{1'b0}});
    wire is_active_smc = all_smc_enable ? 1'b1 : smc_strb_reg[current_smc];
    
    // 状态机
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // 下一状态逻辑 - 修复状态转换条件
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (stb_u_valid) begin
                    next_state = INIT;
                end
            end
            
            INIT: begin
                next_state = AWVALID;
            end
            
            AWVALID: begin
                if (axi_awready) begin
                    next_state = WVALID;
                end
            end
            
            WVALID: begin
                if (axi_wready) begin
                    if (burst_count == brst_reg - 1) begin
                        next_state = WAIT_RESP;
                    end else begin
                        next_state = WVALID; // 保持状态直到所有数据传输完成
                    end
                end
            end
            
            WAIT_RESP: begin
                if (axi_bvalid) begin
                    if (current_smc == SMC_COUNT - 1) begin
                        next_state = DONE;
                    end else begin
                        next_state = INIT; // 回到INIT状态处理下一个SMC
                    end
                end
            end
            
            DONE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // 控制信号和计数器 - 修复控制逻辑
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_smc <= 0;
            burst_count <= 0;
            ur_re <= 0;
            axi_awvalid <= 0;
            axi_wvalid <= 0;
            axi_wlast <= 0;
            axi_bready <= 0;
            stb_d_valid <= 0;
            stb_d_done <= 0;
            
            // 寄存器微指令参数
            smc_strb_reg <= 0;
            byte_strb_reg <= 0;
            brst_reg <= 0;
            gr_base_addr_reg <= 0;
            ur_id_reg <= 0;
            ur_addr_reg <= 0;
        end else begin
            case (current_state)
                IDLE: begin
                    current_smc <= 0;
                    burst_count <= 0;
                    ur_re <= 0;
                    axi_awvalid <= 0;
                    axi_wvalid <= 0;
                    axi_wlast <= 0;
                    axi_bready <= 0;
                    stb_d_valid <= 0;
                    stb_d_done <= 0;
                    
                    // 锁存微指令参数
                    if (stb_u_valid) begin
                        smc_strb_reg <= stb_u_smc_strb;
                        byte_strb_reg <= stb_u_byte_strb;
                        brst_reg <= stb_u_brst;
                        gr_base_addr_reg <= stb_u_gr_base_addr;
                        ur_id_reg <= stb_u_ur_id;
                        ur_addr_reg <= stb_u_ur_addr;
                    end
                end
                
                INIT: begin
                    burst_count <= 0;
                    ur_re <= 1'b1;
                    ur_id <= ur_id_reg;
                    ur_addr <= ur_addr_reg;
                    axi_awvalid <= 1'b1;
                end
                
                AWVALID: begin
                    if (axi_awready) begin
                        axi_awvalid <= 1'b0;
                        axi_wvalid <= 1'b1;
                        if (burst_count == brst_reg - 1) begin
                            axi_wlast <= 1'b1;
                        end
                    end
                end
                
                WVALID: begin
                    if (axi_wready) begin
                        if (burst_count == brst_reg - 1) begin
                            axi_wvalid <= 1'b0;
                            axi_wlast <= 1'b0;
                            axi_bready <= 1'b1;
                            burst_count <= 0;
                        end else begin
                            burst_count <= burst_count + 1;
                            ur_addr <= ur_addr_reg + burst_count + 1; // 预取下一个地址
                            if (burst_count == brst_reg - 2) begin
                                axi_wlast <= 1'b1;
                            end
                        end
                    end
                end
                
                WAIT_RESP: begin
                    if (axi_bvalid) begin
                        axi_bready <= 1'b0;
                        if (current_smc == SMC_COUNT - 1) begin
                            stb_d_valid <= 1'b1;
                            stb_d_done <= 1'b1;
                        end else begin
                            current_smc <= current_smc + 1;
                        end
                    end
                end
                
                DONE: begin
                    stb_d_valid <= 1'b0;
                    stb_d_done <= 1'b0;
                    current_smc <= 0;
                end
            endcase
        end
    end

endmodule