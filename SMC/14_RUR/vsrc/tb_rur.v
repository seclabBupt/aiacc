module RUR_tb;

    // ---------- DUT ----------
    logic        clk;
    logic        rst_n;
    logic [96:0] cru_rur;
    logic [127:0] dr_rur_d;

    localparam LOCAL_SMC_ID = 5'd3;      // 与 RTL 的 parameter 一致
    RUR #(.LOCAL_SMC_ID(LOCAL_SMC_ID)) dut (.*);

    // ---------- 时钟 ----------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ---------- 统计 ----------
    int total = 0;
    int pass  = 0;
    int fail  = 0;

    // ---------- 随机种子 ----------
    initial begin
        int seed;
        if (!$value$plusargs("SEED=%d", seed)) seed = 32'hdead_beef;
        $srandom(seed);
        $display("[TB] Random seed = %0d", seed);
    end

    // ---------- 初始化 UR-RAM ----------
    task automatic init_ur_ram();
        for (int b = 0; b < 8; b++) begin
            for (int a = 0; a < 256; a++) begin
                @(posedge clk);
                // 用4个32位随机数拼接成128位
                dut.ur_ram[b][a] = {$urandom, $urandom, $urandom, $urandom};
            end
        end
        $display("[TB] Initialized UR-RAM with random data.");
    endtask

    // ---------- 随机指令生成 ----------
    task automatic issue_cmd(input bit valid,
                             input bit bad_id,
                             input bit corner);
        logic [96:0] cmd = '0;
        int pick;
        cmd[96]      = valid;
        cmd[95:91]   = bad_id ? ($urandom & 5'h1F) : LOCAL_SMC_ID;
        cmd[90:88]   = $urandom_range(7);
        cmd[87:80]   = $urandom_range(255);

        // 16 组 {4b addr, 1b vld}
        pick = $urandom_range(15);
        for (int i = 0; i < 16; i++) begin
            cmd[79 - i*5 -: 4] = $urandom_range(15);             // addr
            cmd[79 - i*5 - 4]  = (i == pick) || corner ? 1'b1 : 1'b0; // vld
        end
        cru_rur = cmd;
    endtask

    // ---------- 期望模型 ----------
    logic [127:0] ref_dr_rur = 128'h0;

    task automatic update_ref();
        automatic logic [127:0] cur_mask = 128'h0;
        if (cru_rur[96] && (cru_rur[95:91] == LOCAL_SMC_ID)) begin
            automatic logic [127:0] raw;
            raw = dut.ur_ram[cru_rur[90:88]][cru_rur[87:80]];
            for (int x = 0; x < 16; x++) begin
                if (cru_rur[79 - x*5 - 4]) begin
                    automatic int sel = cru_rur[79 - x*5 -: 4];
                    cur_mask[8*x +: 8] = raw[8*sel +: 8];
                end
            end
            ref_dr_rur |= cur_mask;
        end
    endtask

    // --------------------------------------------------
    //  替换原来的 check_result 任务
    // --------------------------------------------------
    task automatic check_result();
        total++;
        if (dr_rur_d === ref_dr_rur) begin
            pass++;
            $display("[PASS] t=%0t exp=%h act=%h", $time, ref_dr_rur, dr_rur_d);
        end else begin
            fail++;
            $display("[FAIL] t=%0t exp=%h act=%h", $time, ref_dr_rur, dr_rur_d);
        end

        // ===== 新增：打印本次指令的所有字段 =====
        $display("    CMD     = %097b", cru_rur);                 // 97 位原始指令
        $display("    vld     = %b", cru_rur[96]);
        $display("    smc_id  = %b (%0d)", cru_rur[95:91], cru_rur[95:91]);
        $display("    ur_id   = %b (%0d)", cru_rur[90:88], cru_rur[90:88]);
        $display("    ur_addr = %b (0x%02h)", cru_rur[87:80], cru_rur[87:80]);

        // 16 组 {4b addr, 1b vld}
        for (int i = 0; i < 16; i++) begin
            logic [3:0] lo_addr = cru_rur[79 - i*5 -: 4];
            logic       lo_vld  = cru_rur[79 - i*5 - 4];
            $display("    lo[%02d]  addr=%b (%0d)  vld=%b",
                    i, lo_addr, lo_addr, lo_vld);
        end

        // ===== 替换原来的 RAM_RAW 打印 =====
        // 先把 128-bit 原始数据抓出来，再分 4 个 32-bit 段打印，确保无截断
        begin
            logic [127:0] ram_raw;
            int           _bank = cru_rur[10:8];   // 3-bit
            int           _addr = cru_rur[7:0];    // 8-bit

            ram_raw = dut.ur_ram[_bank][_addr];

            $display("=== debug RAM_RAW ==================================");
            $display(" bank=%0d  addr=0x%02h", _bank, _addr);
            $display(" raw[127:96] = 0x%08h", ram_raw[127:96]);            
            $display(" raw[95:64]  = 0x%08h", ram_raw[95:64]);            
            $display(" raw[63:32]  = 0x%08h", ram_raw[63:32]);
            $display(" raw[31:0]   = 0x%08h", ram_raw[31:0]);

            $display(" raw(128'h)  = 128'h%032h",
                    {ram_raw[127:96], ram_raw[95:64], ram_raw[63:32], ram_raw[31:0]});
            $display("===================================================");
        end

    endtask



    // ---------- 主测试序列 ----------
    initial begin
        int nv, nb, nc;
        rst_n   = 0;
        cru_rur = '0;
        ref_dr_rur = '0;
        repeat (3) @(posedge clk);
        rst_n = 1;

        init_ur_ram();

        if (!$value$plusargs("NUM_VALID=%d", nv)) nv = 100;
        if (!$value$plusargs("NUM_BAD=%d",   nb)) nb = 20;
        if (!$value$plusargs("NUM_CORNER=%d",nc)) nc = 10;

        repeat (nv) begin
            @(posedge clk);
            issue_cmd(1, 0, 0);
            @(posedge clk);   // 让 RTL 采样
            update_ref();
            check_result();
        end
        repeat (nb) begin
            @(posedge clk);
            issue_cmd(1, 1, 0);  // smc_id 不匹配
            @(posedge clk);
            update_ref();
            check_result();
        end
        repeat (nc) begin
            @(posedge clk);
            issue_cmd(1, 0, 1);  // corner: 开启多组 vld
            @(posedge clk);
            update_ref();
            check_result();
        end

        @(posedge clk);
        #1;
        $display("\n=========================================");
        $display("TEST FINISHED");
        $display("Total=%0d PASS=%0d FAIL=%0d", total, pass, fail);
        if (fail == 0) $display("ALL TESTS PASSED!");
        else           $display("SOME TESTS FAILED!");
        $display("=========================================\n");
        $finish;
    end


    // ---------- 波形 ----------
    initial begin
        $fsdbDumpfile("tb.vpd");
        $fsdbDumpvars(0, RUR_tb);
    end

endmodule
