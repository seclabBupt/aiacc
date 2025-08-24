`timescale 1ns/1ps

module burst_store_tb;

    // 参数定义
    localparam SMC_COUNT      = 4;      // 简化测试使用4个SMC
    localparam UR_BYTE_CNT    = 16;     // 16字节
    localparam ADDR_WIDTH     = 32;     // 地址宽度
    localparam DATA_WIDTH     = 128;    // 数据宽度
    localparam INTLV_STEP     = 64;     // 地址交错步长
    localparam BURST_WIDTH    = 8;      // Burst长度位宽
    
    // 时钟和复位
    reg clk;
    reg rst_n;
    
    // 输入信号
    reg                         stb_u_valid;
    reg  [SMC_COUNT-1:0]        stb_u_smc_strb;
    reg  [3:0]                  stb_u_byte_strb;
    reg  [BURST_WIDTH-1:0]      stb_u_brst;
    reg  [ADDR_WIDTH-1:0]       stb_u_gr_base_addr;
    reg  [2:0]                  stb_u_ur_id;
    reg  [10:0]                 stb_u_ur_addr;
    
    // SMC寄存器数据
    wire [DATA_WIDTH-1:0]       ur_rdata;
    
    // AXI接口 - 直接连接到内存模型
    wire                        axi_awvalid;
    wire [ADDR_WIDTH-1:0]       axi_awaddr;
    wire                        axi_awready;
    
    wire                        axi_wvalid;
    wire [DATA_WIDTH-1:0]       axi_wdata;
    wire [15:0]                 axi_wstrb;
    wire                        axi_wlast;
    wire                        axi_wready;
    
    wire                        axi_bvalid;
    wire                        axi_bready;
    
    // UR接口
    wire                        ur_re;
    wire [2:0]                  ur_id;
    wire [10:0]                 ur_addr;
    
    // 输出信号
    wire                        stb_d_valid;
    wire                        stb_d_done;
    
    // 测试控制
    integer test_num = 0;
    integer error_count = 0;
    integer result_fd;
    integer src_data_fd;
    integer mem_data_fd;
    
    // 时钟生成
    always #5 clk = ~clk;
    
    // 复位生成
    initial begin
        clk = 0;
        rst_n = 0;
        #20 rst_n = 1;
    end
    
    // UR数据模拟
    reg [DATA_WIDTH-1:0] ur_data [0:7]; // 8个UR
    assign ur_rdata = ur_data[ur_id];
    
    // 生成随机UR数据并写入文件
    integer rand_i;
    initial begin
        result_fd = $fopen("result.txt", "w");
        src_data_fd = $fopen("../sim_output/src_data.txt", "w");
        mem_data_fd = $fopen("../sim_output/mem_data.txt", "w");
        
        if (result_fd == 0 || src_data_fd == 0 || mem_data_fd == 0) begin
            $display("[ERROR] Cannot open output files!");
            $finish;
        end
        
        $fdisplay(result_fd, "==== UR Random Data ====");
        $fdisplay(src_data_fd, "==== Source Data (UR Content) ====");
        $fdisplay(mem_data_fd, "==== Memory Data (After Write) ====");
        
        for (rand_i = 0; rand_i < 8; rand_i = rand_i + 1) begin
            ur_data[rand_i] = $random;
            $fdisplay(result_fd, "ur_data[%0d] = 0x%h", rand_i, ur_data[rand_i]);
            $fdisplay(src_data_fd, "UR[%0d] = 0x%h", rand_i, ur_data[rand_i]);
        end
        
        $fdisplay(result_fd, "========================");
        $fdisplay(src_data_fd, "========================");
        $fdisplay(mem_data_fd, "========================");
    end
    
    // 实例化Burst Store模块
    burst_store #(
        .SMC_COUNT(SMC_COUNT),
        .UR_BYTE_CNT(UR_BYTE_CNT),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .INTLV_STEP(INTLV_STEP),
        .BURST_WIDTH(BURST_WIDTH),
        .UR_ADDR_WIDTH(11),
        .UR_ID_WIDTH(3))
    uut (
        .clk(clk),
        .rst_n(rst_n),
        
        .stb_u_valid(stb_u_valid),
        .stb_u_smc_strb(stb_u_smc_strb),
        .stb_u_byte_strb(stb_u_byte_strb),
        .stb_u_brst(stb_u_brst),
        .stb_u_gr_base_addr(stb_u_gr_base_addr),
        .stb_u_ur_id(stb_u_ur_id),
        .stb_u_ur_addr(stb_u_ur_addr),
        
        .ur_re(ur_re),
        .ur_id(ur_id),
        .ur_addr(ur_addr),
        .ur_rdata(ur_rdata),
        
        .axi_awvalid(axi_awvalid),
        .axi_awaddr(axi_awaddr),
        .axi_awready(axi_awready),
        
        .axi_wvalid(axi_wvalid),
        .axi_wdata(axi_wdata),
        .axi_wstrb(axi_wstrb),
        .axi_wlast(axi_wlast),
        .axi_wready(axi_wready),
        
        .axi_bvalid(axi_bvalid),
        .axi_bready(axi_bready),
        
        .stb_d_valid(stb_d_valid),
        .stb_d_done(stb_d_done)
    );
    
    // 内存模型实例 - 直接连接到burst_store
    axi_mem_model #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .MEM_SIZE(1024 * 1024))
    mem (
        .clk(clk),
        
        .axi_awvalid(axi_awvalid),
        .axi_awaddr(axi_awaddr),
        .axi_awready(axi_awready),
        
        .axi_wvalid(axi_wvalid),
        .axi_wdata(axi_wdata),
        .axi_wstrb(axi_wstrb),
        .axi_wlast(axi_wlast),
        .axi_wready(axi_wready),
        
        .axi_bvalid(axi_bvalid),
        .axi_bready(axi_bready),
        
        .rd_data() // 未使用
    );
    
    // 记录AXI写入数据
    always @(posedge clk) begin
        if (axi_wvalid && axi_wready) begin
            $fdisplay(src_data_fd, "[WRITE] Addr: 0x%h, Data: 0x%h, Strb: 0x%h, Last: %b",
                     axi_awaddr, axi_wdata, axi_wstrb, axi_wlast);
        end
    end
    
    // 初始化
    initial begin
        stb_u_valid = 0;
        stb_u_smc_strb = 0;
        stb_u_byte_strb = 0;
        stb_u_brst = 0;
        stb_u_gr_base_addr = 0;
        stb_u_ur_id = 0;
        stb_u_ur_addr = 0;
        
        // 等待初始化完成
        #30;
        
        // 运行固定测试用例
        run_test(4, 4'h0, 32'h1000, 0, 0);      // 测试1：完整传输，所有SMC开启
        run_test(3, 4'h3, 32'h2000, 1, 0);      // 测试2：部分字节传输，所有SMC开启
        run_test(8, 4'hF, 32'h3000, 2, 0);      // 测试3：高位字节传输，所有SMC开启
        run_test(5, 4'h8, 32'h4000, 3, 0);      // 测试4：低位字节传输，所有SMC开启
        
        // 测试结果
        $display("\n[TEST SUMMARY]");
        $display("Total tests: %0d", test_num);
        $display("Errors: %0d", error_count);
        if (error_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end

        // 写入 result.txt
        $fdisplay(result_fd, "\n[TEST SUMMARY]");
        $fdisplay(result_fd, "Total tests: %0d", test_num);
        $fdisplay(result_fd, "Errors: %0d", error_count);
        if (error_count == 0)
            $fdisplay(result_fd, "ALL TESTS PASSED!");
        else
            $fdisplay(result_fd, "SOME TESTS FAILED!");
        
        // 导出内存内容到文件
        dump_memory_contents();
        
        $fclose(result_fd);
        $fclose(src_data_fd);
        $fclose(mem_data_fd);
        $finish;
    end
    
    // 测试任务
    task run_test;
        input [BURST_WIDTH-1:0] brst;
        input [3:0] byte_strb;
        input [ADDR_WIDTH-1:0] base_addr;
        input [2:0] ur_id;
        input [SMC_COUNT-1:0] smc_strb;   // SMC使能掩码
        
        integer timeout;
        integer local_error;
        integer j, k;
        reg [ADDR_WIDTH-1:0] calc_addr;
    begin
        test_num = test_num + 1;
        local_error = 0;
        timeout = 0;
        
        $display("\n[TEST %0d] Starting test: brst=%0d, byte_strb=0x%h, base_addr=0x%h, ur_id=%0d, smc_strb=0b%b",
                 test_num, brst, byte_strb, base_addr, ur_id, smc_strb);
        $fdisplay(result_fd, "\n[TEST %0d] brst=%0d, byte_strb=0x%h, base_addr=0x%h, ur_id=%0d, smc_strb=0b%b",
                 test_num, brst, byte_strb, base_addr, ur_id, smc_strb);
        $fdisplay(src_data_fd, "\n[TEST %0d] brst=%0d, byte_strb=0x%h, base_addr=0x%h, ur_id=%0d, smc_strb=0b%b",
                 test_num, brst, byte_strb, base_addr, ur_id, smc_strb);
        
        // 设置测试参数
        stb_u_valid = 1;
        stb_u_smc_strb = smc_strb;
        stb_u_byte_strb = byte_strb;
        stb_u_brst = brst;
        stb_u_gr_base_addr = base_addr;
        stb_u_ur_id = ur_id;
        stb_u_ur_addr = 0;
        
        // 记录预期写入的数据
        $fdisplay(src_data_fd, "[EXPECTED DATA] UR[%0d] = 0x%h", ur_id, ur_data[ur_id]);
        
        // 等待测试完成或超时
        while (!stb_d_done && timeout < 1000) begin
            #10;
            timeout = timeout + 1;
        end
        
        if (timeout >= 1000) begin
            $display("[ERROR] Test %0d timed out!", test_num);
            $fdisplay(result_fd, "[ERROR] Test %0d timed out!", test_num);
            $fdisplay(src_data_fd, "[ERROR] Test %0d timed out!", test_num);
            error_count = error_count + 1;
            local_error = local_error + 1;
        end else begin
            $display("[TEST %0d] Completed successfully", test_num);
            $fdisplay(result_fd, "[TEST %0d] Completed successfully", test_num);
            $fdisplay(src_data_fd, "[TEST %0d] Completed successfully", test_num);
            
            // 记录实际写入的内存数据
            $fdisplay(mem_data_fd, "\n[TEST %0d] Memory Contents:", test_num);
            for (j = 0; j < SMC_COUNT; j = j + 1) begin
                // 跳过未使能的SMC
                if (smc_strb != 0 && !smc_strb[j]) continue;
                
                for (k = 0; k < brst; k = k + 1) begin
                    calc_addr = base_addr + (j * INTLV_STEP) + (k * UR_BYTE_CNT);
                    $fdisplay(mem_data_fd, "Addr: 0x%h, Data: 0x%h", calc_addr, mem.memory[calc_addr]);
                end
            end
        end
        
        // 复位信号
        stb_u_valid = 0;
        #100;
    end
    endtask
    
    // 导出内存内容到文件
    task dump_memory_contents;
        integer i, start_addr, end_addr;
        reg [ADDR_WIDTH-1:0] addr;
    begin
        // 确定导出范围：所有测试使用的地址范围
        start_addr = 32'h1000;
        end_addr = 32'h5000;
        
        $fdisplay(mem_data_fd, "\n==== Full Memory Dump from 0x%h to 0x%h ====", start_addr, end_addr);
        for (addr = start_addr; addr <= end_addr; addr = addr + 16) begin
            if (mem.memory[addr] !== 'hx) begin
                $fdisplay(mem_data_fd, "Addr: 0x%h, Data: 0x%h", addr, mem.memory[addr]);
            end
        end
        $fdisplay(mem_data_fd, "==== End of Memory Dump ====");
    end
    endtask
    
    // 生成波形
    initial begin
        $dumpfile("burst_store.vcd");
        $dumpvars(0, burst_store_tb);
    end

endmodule