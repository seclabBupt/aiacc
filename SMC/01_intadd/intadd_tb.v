`timescale 1ns / 1ps

module intadd_tb;

    // === DPI-C接口声明 ===
    import "DPI-C" function void add32_128bit(
        input longint unsigned src0_high, input longint unsigned src0_low,
        input longint unsigned src1_high, input longint unsigned src1_low,
        input int sign_s0, input int sign_s1,
        output longint unsigned dst_high, output longint unsigned dst_low,
        output longint unsigned st_high, output longint unsigned st_low
    );
    import "DPI-C" function void add8_128bit(
        input longint unsigned src0_high, input longint unsigned src0_low,
        input longint unsigned src1_high, input longint unsigned src1_low,
        input longint unsigned src2_high, input longint unsigned src2_low,
        input int sign_s0, input int sign_s1, input int sign_s2,
        output longint unsigned dst0_high, output longint unsigned dst0_low,
        output longint unsigned dst1_high, output longint unsigned dst1_low,
        output longint unsigned st_high, output longint unsigned st_low
    );

    reg [127:0] src_reg0, src_reg1, src_reg2;
    reg [1:0]   precision_s0, precision_s1, precision_s2;
    reg         sign_s0, sign_s1, sign_s2;
    reg         inst_valid, update_st;
    reg         clk, rst_n;
    reg [10:0]  cru_intadd; // 微指令控制信号

    wire [127:0] dst_reg0, dst_reg1, st;

    // 时钟
    always #5 clk = ~clk;

    // DUT
    intadd uut (
        .clk(clk),
        .rst_n(rst_n),
        .src_reg0(src_reg0),
        .src_reg1(src_reg1),
        .src_reg2(src_reg2),
        .cru_intadd(cru_intadd),
        .dst_reg0(dst_reg0),
        .dst_reg1(dst_reg1),
        .st(st)
    );

    integer result_file;
    integer pass_cnt, total_cnt;

    // 初始化
    task init_inputs;
    begin
        src_reg0 = 0; src_reg1 = 0; src_reg2 = 0;
        precision_s0 = 0; precision_s1 = 0; precision_s2 = 0;
        sign_s0 = 0; sign_s1 = 0; sign_s2 = 0;
        inst_valid = 0; update_st = 0;
        cru_intadd = 11'd0;
    end
    endtask

    // === compare_and_log_32bit ===
    function automatic int compare_and_log_32bit(
        input [127:0] rtl_dst,
        input [127:0] rtl_st,
        input [127:0] src0,
        input [127:0] src1,
        input bit s0,
        input bit s1
    );
        longint unsigned c_dst_high, c_dst_low;
        longint unsigned c_st_high, c_st_low;
        integer fh;
        begin
            // 调用参考模型
            add32_128bit(src0[127:64], src0[63:0],
                         src1[127:64], src1[63:0],
                         s0, s1,
                         c_dst_high, c_dst_low,
                         c_st_high, c_st_low);

            // 比较
            if ({c_dst_high, c_dst_low} !== rtl_dst ||
                {c_st_high, c_st_low}   !== rtl_st) begin
                fh = $fopen("result.txt", "a");
                $fdisplay(fh, "[FAIL][32bit] src0=%h src1=%h RTL_dst=%h C_dst=%h RTL_st=%h C_st=%h",
                               src0, src1, rtl_dst, {c_dst_high, c_dst_low},
                               rtl_st, {c_st_high, c_st_low});
                $fclose(fh);
                compare_and_log_32bit = 0;
            end else begin
                compare_and_log_32bit = 1;
            end
        end
    endfunction

    // === compare_and_log_4_8bit ===
    function automatic int compare_and_log_4_8bit(
        input [127:0] rtl_dst0,
        input [127:0] rtl_dst1,
        input [127:0] rtl_st,
        input [127:0] src0,
        input [127:0] src1,
        input [127:0] src2,
        input bit s0,
        input bit s1,
        input bit s2
    );
        longint unsigned c_dst0_high, c_dst0_low;
        longint unsigned c_dst1_high, c_dst1_low;
        longint unsigned c_st_high, c_st_low;
        integer fh;
        begin
            // 调用参考模型
            add8_128bit(src0[127:64], src0[63:0],
                        src1[127:64], src1[63:0],
                        src2[127:64], src2[63:0],
                        s0, s1, s2,
                        c_dst0_high, c_dst0_low,
                        c_dst1_high, c_dst1_low,
                        c_st_high, c_st_low);

            // 比较
            if ({c_dst0_high, c_dst0_low} !== rtl_dst0 ||
                {c_dst1_high, c_dst1_low} !== rtl_dst1 ||
                {c_st_high, c_st_low}     !== rtl_st) begin
                fh = $fopen("result.txt", "a");
                $fdisplay(fh, "[FAIL][4+8bit] src0=%h src1=%h src2=%h RTL_dst0=%h C_dst0=%h RTL_dst1=%h C_dst1=%h RTL_st=%h C_st=%h",
                               src0, src1, src2,
                               rtl_dst0, {c_dst0_high, c_dst0_low},
                               rtl_dst1, {c_dst1_high, c_dst1_low},
                               rtl_st, {c_st_high, c_st_low});
                $fclose(fh);
                compare_and_log_4_8bit = 0;
            end else begin
                compare_and_log_4_8bit = 1;
            end
        end
    endfunction

    // === 测试流程 ===
    integer i;
    initial begin
        clk = 0;
        rst_n = 0;
        pass_cnt = 0;
        total_cnt = 0;

        result_file = $fopen("result.txt", "w");
        $fdisplay(result_file, "==== INTADD TB TEST START ====");
        $fclose(result_file);

        init_inputs;
        #10 rst_n = 1;

        // 测试1: 32bit 正溢出
        src_reg0 = {32'h7FFFFFFF, 32'h7FFFFFFF, 32'h7FFFFFFF, 32'h7FFFFFFF};
        src_reg1 = {32'h00000001, 32'h00000001, 32'h00000001, 32'h00000001};
        precision_s0 = 2'b11; precision_s1 = 2'b11; precision_s2 = 2'b11;
        sign_s0 = 1; sign_s1 = 1; sign_s2 = 0;
        inst_valid = 1; update_st = 1;
        cru_intadd = {inst_valid, precision_s0, precision_s1, precision_s2,
                      sign_s0, sign_s1, sign_s2, update_st};
        #10;
        if (compare_and_log_32bit(dst_reg0, st, src_reg0, src_reg1, sign_s0, sign_s1))
            pass_cnt++;
        total_cnt++;

        // 测试2: 32bit 负溢出
        src_reg0 = {32'h80000000, 32'h80000000, 32'h80000000, 32'h80000000};
        src_reg1 = {32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF};
        precision_s0 = 2'b11; precision_s1 = 2'b11; precision_s2 = 2'b11;
        sign_s0 = 1; sign_s1 = 1; sign_s2 = 0;
        inst_valid = 1; update_st = 1;
        cru_intadd = {inst_valid, precision_s0, precision_s1, precision_s2,
                      sign_s0, sign_s1, sign_s2, update_st};
        #10;
        if (compare_and_log_32bit(dst_reg0, st, src_reg0, src_reg1, sign_s0, sign_s1))
            pass_cnt++;
        total_cnt++;

        // 测试3: 4+8bit 随机
        for (i = 0; i < 1000; i++) begin
            src_reg0 = {$random, $random, $random, $random};
            src_reg1 = {$random, $random, $random, $random};
            src_reg2 = {$random, $random, $random, $random};
            precision_s0 = 2'b00; precision_s1 = 2'b00; precision_s2 = 2'b00;
            sign_s0 = $random & 1; sign_s1 = $random & 1; sign_s2 = $random & 1;
            inst_valid = 1; update_st = 0;
            cru_intadd = {inst_valid, precision_s0, precision_s1, precision_s2,
                          sign_s0, sign_s1, sign_s2, update_st};
            #10;
            if (compare_and_log_4_8bit(dst_reg0, dst_reg1, st, src_reg0, src_reg1, src_reg2,
                                       sign_s0, sign_s1, sign_s2))
                pass_cnt++;
            total_cnt++;
        end

        // 结果
        result_file = $fopen("result.txt", "a");
        $fdisplay(result_file, "==== SUMMARY ====");
        $fdisplay(result_file, "PASS: %0d / TOTAL: %0d", pass_cnt, total_cnt);
        $fdisplay(result_file, "==== END ====");
        $fclose(result_file);

        $display("All tests completed. PASS: %0d / TOTAL: %0d", pass_cnt, total_cnt);
        $finish;
    end

endmodule
