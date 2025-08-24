`timescale 1ns/1ps

module axi_mem_model #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 128,
    parameter MEM_SIZE = 1024 * 1024 // 1MB内存
)(
    // 时钟输入
    input  wire                  clk,
    
    // AXI 从接口
    input  wire                  axi_awvalid,
    input  wire [ADDR_WIDTH-1:0] axi_awaddr,
    output reg                   axi_awready,
    
    input  wire                  axi_wvalid,
    input  wire [DATA_WIDTH-1:0] axi_wdata,
    input  wire [15:0]           axi_wstrb,
    input  wire                  axi_wlast,
    output reg                   axi_wready,
    
    output reg                   axi_bvalid,
    input  wire                  axi_bready,
    
    // 内存读取接口
    output reg [DATA_WIDTH-1:0]  rd_data
);

    // 内存数组
    reg [DATA_WIDTH-1:0] memory [0:MEM_SIZE-1];
    
    // 内部状态
    reg [ADDR_WIDTH-1:0] awaddr_latched;
    reg aw_received;
    reg w_received;
    reg [7:0] burst_counter;
    
    // 初始化
    initial begin
        axi_awready = 1'b1;  // 初始化为就绪
        axi_wready = 1'b1;   // 初始化为就绪
        axi_bvalid = 1'b0;
        awaddr_latched = 0;
        aw_received = 1'b0;
        w_received = 1'b0;
        burst_counter = 0;
        rd_data = 0;
    end
    
    // AWREADY 生成 - 简化模型，总是准备好
    always @(*) begin
        axi_awready = !aw_received;
    end
    
    // WREADY 生成 - 简化模型，总是准备好
    always @(*) begin
        axi_wready = !w_received;
    end
    
// 在内存模型中修改地址计算
always @(posedge clk) begin
    // 锁存地址
    if (axi_awvalid && axi_awready) begin
        // 确保地址对齐到16字节边界
        awaddr_latched <= {axi_awaddr[ADDR_WIDTH-1:4], 4'b0};
        aw_received <= 1'b1;
        axi_awready <= 1'b0;
        burst_counter <= 0;
    end
    
    // 写入数据
    if (axi_wvalid && axi_wready) begin
        // 计算实际内存地址
        reg [ADDR_WIDTH-1:0] actual_addr;
        actual_addr = awaddr_latched + (burst_counter * 16); // 每个burst增加16字节
        
        for (integer i = 0; i < 16; i = i + 1) begin
            if (axi_wstrb[i]) begin
                memory[actual_addr][i*8 +: 8] <= axi_wdata[i*8 +: 8];
            end
        end
        burst_counter <= burst_counter + 1;
        
        // 如果是最后一个数据，准备发送响应
        if (axi_wlast) begin
            w_received <= 1'b1;
            axi_bvalid <= 1'b1;
        end
    end
    
    // 响应完成
    if (axi_bvalid && axi_bready) begin
        axi_bvalid <= 1'b0;
        aw_received <= 1'b0;
        w_received <= 1'b0;
        axi_awready <= 1'b1;
        axi_wready <= 1'b1;
    end
end
    // 读取操作 - 组合逻辑
    always @(*) begin
        rd_data = memory[awaddr_latched]; // 读取当前锁存的地址
    end

endmodule