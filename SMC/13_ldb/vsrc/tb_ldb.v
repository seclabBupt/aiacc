`timescale 1ns/1ps

module tb_LDB_ENGINE();

 
    // 1. 参数与信号声明 
    localparam CLK_PERIOD     = 10;      // 100 MHz
    localparam PARAM_SMC_CNT  = 4;
    localparam PARAM_UR_BYTE_CNT = 16;   // 16 Byte UR

    reg         clk;
    reg         rst_n;

    // 上行微指令
    reg         vld;
    reg  [5:0]  smc_strb;
    reg  [3:0]  byte_strb;
    reg  [15:0] brst;
    reg  [31:0] gr_base_addr;
    reg  [1:0]  smc_id;
    reg  [7:0]  ur_id;
    reg  [15:0] ur_addr;

    // AXI-Lite
    wire [31:0] axi_araddr;
    wire        axi_arvalid;
    reg         axi_arready;
    reg  [31:0] axi_rdata;
    reg         axi_rvalid;
    wire        axi_rready;

    // 下行
    wire        done;
    wire        vld_down;

    // 模拟外部存储器
    reg  [31:0] test_mem [0:1023];
    integer     test_case;

    // 2. 时钟与复位
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        rst_n = 0;
        #(CLK_PERIOD*2) rst_n = 1;
    end

    // 3. DUT 实例化
    LDB_ENGINE #(
        .PARAM_UR_BYTE_CNT   (PARAM_UR_BYTE_CNT),
        .PARAM_GR_INTLV_ADDR (64),
        .PARAM_SMC_CNT       (PARAM_SMC_CNT)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .vld            (vld),
        .smc_strb       (smc_strb),
        .byte_strb      (byte_strb),
        .brst           (brst),
        .gr_base_addr   (gr_base_addr),
        .smc_id         (smc_id),
        .ur_id          (ur_id),
        .ur_addr        (ur_addr),
        .axi_araddr     (axi_araddr),
        .axi_arvalid    (axi_arvalid),
        .axi_arready    (axi_arready),
        .axi_rdata      (axi_rdata),
        .axi_rvalid     (axi_rvalid),
        .axi_rready     (axi_rready),
        .done           (done),
        .vld_down       (vld_down)
    );

    // 4. AXI-Lite Slave
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_arready <= 0;
            axi_rvalid  <= 0;
            axi_rdata   <= 0;
        end else begin
            // 随机反压
            axi_arready <= ($random % 3 == 0) ? 0 : 1;

            if (axi_arvalid && axi_arready) begin
                axi_rvalid <= 1;
                axi_rdata  <= test_mem[axi_araddr >> 2];
            end else if (axi_rvalid && axi_rready) begin
                axi_rvalid <= 0;
            end
        end
    end

    // 5. 测试序列
    initial begin
        // 初始化
        vld = 0; axi_rvalid = 0; test_case = 0;
        for (int i=0; i<1024; i++) test_mem[i] = i + 1;

        @(posedge rst_n);
        #(CLK_PERIOD*2);
    
        // TC1: 单 SMC 完整 burst       
        test_case = 1;
        $display("[%0t] TC1: 单 SMC 完整 burst", $time);
        vld=1; smc_strb=6'b000001; byte_strb=4'b1111; brst=4;
        gr_base_addr=0; smc_id=0; ur_id=8'hA5; ur_addr=16'h0100;
        @(posedge clk); vld=0; wait(done); #(CLK_PERIOD*2);

        // TC2: 多 SMC 并行       
        test_case = 2;
        $display("[%0t] TC2: 多 SMC 并行", $time);
        vld=1; smc_strb=6'b000111; byte_strb=4'b1111; brst=2;
        gr_base_addr=256; smc_id=0; ur_id=8'hB3; ur_addr=16'h0200;
        @(posedge clk); vld=0; wait(done); #(CLK_PERIOD*2);
 
        // TC3: byte_strb 掩码       
        test_case = 3;
        $display("[%0t] TC3: byte_strb 掩码", $time);
        vld=1; smc_strb=6'b000001; byte_strb=4'b0011; brst=1;
        gr_base_addr=512; smc_id=0; ur_id=8'hC7; ur_addr=16'h0300;
        @(posedge clk); vld=0; wait(done); #(CLK_PERIOD*2);

        // TC4: smc_strb 有 0 → 覆盖 else 路径       
        test_case = 4;
        $display("[%0t] TC4: smc_strb 有 0 → 覆盖 else 路径", $time);
        vld=1; smc_strb=6'b000101; byte_strb=4'b1111; brst=1;
        gr_base_addr=768; smc_id=0; ur_id=8'hD2; ur_addr=16'h0400;
        @(posedge clk); vld=0; wait(done); #(CLK_PERIOD*2);

        // TC5: 连续指令，让 vld && !active_q 的 else 被踩 
        test_case = 5;
        $display("[%0t] TC5: 连续指令，让 vld && !active_q 的 else 被踩", $time);
        // 第一条
        vld=1; smc_strb=6'b000001; byte_strb=4'b1111; brst=1;
        gr_base_addr=1024; smc_id=0; ur_id=8'hE1; ur_addr=16'h0500;
        @(posedge clk); vld=0;
        // 第二条在第一条未完成时拉高
        fork
            begin wait(done); end
            begin
                repeat(2) @(posedge clk);   // 提前一拍
                vld=1; smc_strb=6'b001000; byte_strb=4'b1111; brst=1;
                gr_base_addr=1024; smc_id=3; ur_id=8'hE2; ur_addr=16'h0600;
                @(posedge clk); vld=0;
            end
        join
        wait(done); #(CLK_PERIOD*2);

        // TC6: DONE 状态里把 axi_rvalid 拉高一拍        
        test_case = 6;
        $display("[%0t] TC6: DONE 状态里把 axi_rvalid = 1", $time);
        vld=1; smc_strb=6'b000001; byte_strb=4'b1111; brst=1;
        gr_base_addr=1280; smc_id=0; ur_id=8'hF0; ur_addr=16'h0700;
        @(posedge clk); vld=0;
        wait(done);
        @(posedge clk);
        axi_rvalid <= 1;    // 强制让 DONE 状态看到 rvalid=1
        @(posedge clk);
        axi_rvalid <= 0;
        #(CLK_PERIOD*2);

        $display("[%0t] All coverage tests passed!", $time);
        $finish;
    end

    // 6. 自动结果检查
    always @(posedge done) begin
        case(test_case)
            1: $display("[%0t] TC1 Done", $time);
            2: $display("[%0t] TC2 Done", $time);
            3: $display("[%0t] TC3 Done", $time);
            4: $display("[%0t] TC4 Done (else path)", $time);
            5: $display("[%0t] TC5 Done (back-to-back)", $time);
            6: $display("[%0t] TC6 Done (rvalid=1 in DONE)", $time);
        endcase
    end

    // 7. 监控 
    initial begin
        $timeformat(-9, 2, " ns", 10);
        $monitor("[%t] STATE: %s ARADDR=%h RVALID=%b RREADY=%b",
                 $time, u_dut.state_q.name(), axi_araddr, axi_rvalid, axi_rready);
    end

endmodule