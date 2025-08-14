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

    // 辅助：将64bit部分和64bit部分合并成128位（用于参考模型输出）
    function automatic [127:0] combine_128;
        input longint unsigned high;
        input longint unsigned low;
        begin
            combine_128 = {high, low};
        end
    endfunction

    // === 辅助：记录 32-bit 各 lane 详细计算过程 ===
    task automatic log_32bit_detailed(
        input integer fh,
        input [127:0] src0,
        input [127:0] src1,
        input bit s0,
        input bit s1,
        input [127:0] rtl_dst,
        input [127:0] rtl_st,
        input [127:0] c_dst,
        input [127:0] c_st
    );
        integer i;
        reg [31:0] a, b, r_rtl, r_c;
        reg [32:0] sum_u;
        reg signed [32:0] sum_s;
        reg signed [31:0] sa, sb, sr;
        reg carry_out, sgn_ovf;
        begin
            $fdisplay(fh, "  Detailed 32-bit lanes at time %0t:", $time);
            for (i = 0; i < 4; i = i + 1) begin
                a = src0[127-32*i -: 32];
                b = src1[127-32*i -: 32];
                r_rtl = rtl_dst[127-32*i -: 32];
                r_c   = c_dst[127-32*i -: 32];

                // 无符号求和用于 carry
                sum_u = ({1'b0, a} + {1'b0, b});
                carry_out = sum_u[32];

                // 有符号求和用于溢出
                sa = a;
                sb = b;
                sum_s = sa + sb; // 33 bits signed
                sr = sum_s[31:0];
                sgn_ovf = (sa[31] == sb[31]) && (sr[31] != sa[31]);

                $fdisplay(fh, "    lane %0d: a=%h (%0d), b=%h (%0d)", i, a, a, b, b);
                $fdisplay(fh, "      unsigned_sum(full) = %0h, carry_out=%0b, truncated_result=%h",
                          sum_u, carry_out, sum_u[31:0]);
                $fdisplay(fh, "      signed_sum(full) = %0h, signed_truncated=%0h, signed_overflow=%0b",
                          sum_s, sr, sgn_ovf);
                $fdisplay(fh, "      RTL_result=%h, C_model_result=%h", r_rtl, r_c);
            end
            $fdisplay(fh, "  RTL_st = %h, C_model_st = %h", rtl_st, c_st);
            $fdisplay(fh, "----------------------------------------------------------------");
        end
    endtask

    // === 辅助：记录 8-bit / 4-bit 各 lane 详细计算过程 ===
    task automatic log_8_4bit_detailed(
        input integer fh,
        input [127:0] src0,
        input [127:0] src1,
        input [127:0] src2,
        input bit s0,
        input bit s1,
        input bit s2,
        input [127:0] rtl_dst0,
        input [127:0] rtl_dst1,
        input [127:0] rtl_st,
        input [127:0] c_dst0,
        input [127:0] c_dst1,
        input [127:0] c_st
    );
        integer i;
        reg [7:0] a8, b8, c8, r0_rtl, r1_rtl, r0_c, r1_c;
        reg [9:0] sum8_u; // 8+8+8 -> up to 10 bits
        reg signed [9:0] sum8_s;
        reg carry8;
        reg signed [7:0] sa8, sb8, sc8, sr8;

        reg [3:0] a4, b4, c4, r4_rtl0, r4_rtl1, r4_c0, r4_c1;
        reg [6:0] sum4_u; // 3*4 = 6 bits enough
        reg signed [6:0] sum4_s;
        reg carry4;
        reg signed [3:0] sa4, sb4, sc4, sr4;
        begin
            $fdisplay(fh, "  Detailed 8-bit lanes at time %0t:", $time);
            for (i = 0; i < 16; i = i + 1) begin
                a8 = src0[127-8*i -: 8];
                b8 = src1[127-8*i -: 8];
                c8 = src2[127-8*i -: 8];

                r0_rtl = rtl_dst0[127-8*i -: 8];
                r1_rtl = rtl_dst1[127-8*i -: 8];
                r0_c   = c_dst0[127-8*i -: 8];
                r1_c   = c_dst1[127-8*i -: 8];

                sum8_u = {2'b00, a8} + {2'b00, b8} + {2'b00, c8};
                carry8 = sum8_u[9];

                sa8 = a8; sb8 = b8; sc8 = c8;
                sum8_s = sa8 + sb8 + sc8;
                sr8 = sum8_s[7:0];

                $fdisplay(fh, "    byte %0d: a=%h (%0d), b=%h (%0d), c=%h (%0d)", i, a8, a8, b8, b8, c8, c8);
                $fdisplay(fh, "      unsigned_sum(full) = %0h, carry_out=%0b, truncated_result=%h",
                          sum8_u, carry8, sum8_u[7:0]);
                $fdisplay(fh, "      signed_sum(full) = %0h, signed_truncated=%0h",
                          sum8_s, sr8);
                $fdisplay(fh, "      RTL_dst0=%h, C_dst0=%h, RTL_dst1=%h, C_dst1=%h",
                          r0_rtl, r0_c, r1_rtl, r1_c);
            end

            $fdisplay(fh, "----------------------------------------------------------------");
            $fdisplay(fh, "  Detailed 4-bit lanes at time %0t:", $time);
            for (i = 0; i < 32; i = i + 1) begin
                a4 = src0[127-4*i -: 4];
                b4 = src1[127-4*i -: 4];
                c4 = src2[127-4*i -: 4];

                // 对 4-bit 小段也打印 RTL/C 模型在对应字节/半字节中的结果（近似匹配）
                r4_rtl0 = rtl_dst0[127-4*i -: 4];
                r4_rtl1 = rtl_dst1[127-4*i -: 4];
                r4_c0   = c_dst0[127-4*i -: 4];
                r4_c1   = c_dst1[127-4*i -: 4];

                sum4_u = {1'b0, a4} + {1'b0, b4} + {1'b0, c4};
                carry4 = sum4_u[6];

                sa4 = a4; sb4 = b4; sc4 = c4;
                sum4_s = sa4 + sb4 + sc4;
                sr4 = sum4_s[3:0];

                $fdisplay(fh, "    nibble %0d: a=%h (%0d), b=%h (%0d), c=%h (%0d)", i, a4, a4, b4, b4, c4, c4);
                $fdisplay(fh, "      unsigned_sum(full) = %0h, carry_out=%0b, truncated_result=%h",
                          sum4_u, carry4, sum4_u[3:0]);
                $fdisplay(fh, "      signed_sum(full) = %0h, signed_truncated=%0h",
                          sum4_s, sr4);
                $fdisplay(fh, "      RTL_dst0_nibble=%h, C_dst0_nibble=%h, RTL_dst1_nibble=%h, C_dst1_nibble=%h",
                          r4_rtl0, r4_c0, r4_rtl1, r4_c1);
            end

            $fdisplay(fh, "  RTL_st = %h, C_model_st = %h", rtl_st, c_st);
            $fdisplay(fh, "================================================================");
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
        reg [127:0] c_dst, c_st_reg;
        integer fh;
        begin
            // 调用参考模型
            add32_128bit(src0[127:64], src0[63:0],
                         src1[127:64], src1[63:0],
                         s0, s1,
                         c_dst_high, c_dst_low,
                         c_st_high, c_st_low);
            c_dst = combine_128(c_dst_high, c_dst_low);
            c_st_reg = combine_128(c_st_high, c_st_low);

            fh = $fopen("result.txt", "a");
            $fdisplay(fh, "[TEST][32bit] time=%0t src0=%h src1=%h sign_s0=%0b sign_s1=%0b", $time, src0, src1, s0, s1);

            // 记录详细计算过程
            log_32bit_detailed(fh, src0, src1, s0, s1, rtl_dst, rtl_st, c_dst, c_st_reg);

            // 比较
            if (c_dst !== rtl_dst || c_st_reg !== rtl_st) begin
                $fdisplay(fh, "[FAIL][32bit] RTL_dst=%h C_dst=%h RTL_st=%h C_st=%h",
                          rtl_dst, c_dst, rtl_st, c_st_reg);
                compare_and_log_32bit = 0;
            end else begin
                $fdisplay(fh, "[PASS][32bit] RTL matches C model for this vector.");
                compare_and_log_32bit = 1;
            end
            $fclose(fh);
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
        reg [127:0] c_dst0, c_dst1, c_st_reg;
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
            c_dst0 = combine_128(c_dst0_high, c_dst0_low);
            c_dst1 = combine_128(c_dst1_high, c_dst1_low);
            c_st_reg = combine_128(c_st_high, c_st_low);

            fh = $fopen("result.txt", "a");
            $fdisplay(fh, "[TEST][4+8bit] time=%0t src0=%h src1=%h src2=%h sign_s=[%0b,%0b,%0b]",
                      $time, src0, src1, src2, s0, s1, s2);

            // 记录详细计算过程（8-bit 和 4-bit 切片）
            log_8_4bit_detailed(fh, src0, src1, src2, s0, s1, s2,
                                rtl_dst0, rtl_dst1, rtl_st,
                                c_dst0, c_dst1, c_st_reg);

            // 比较
            if (c_dst0 !== rtl_dst0 || c_dst1 !== rtl_dst1 || c_st_reg !== rtl_st) begin
                $fdisplay(fh, "[FAIL][4+8bit] RTL_dst0=%h C_dst0=%h RTL_dst1=%h C_dst1=%h RTL_st=%h C_st=%h",
                          rtl_dst0, c_dst0, rtl_dst1, c_dst1, rtl_st, c_st_reg);
                compare_and_log_4_8bit = 0;
            end else begin
                $fdisplay(fh, "[PASS][4+8bit] RTL matches C model for this vector.");
                compare_and_log_4_8bit = 1;
            end
            $fclose(fh);
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
        $fdisplay(result_file, "==== INTADD TB TEST START ====\n(详细的运算过程、每个 lane 的中间值、符号/无符号求和、进位、溢出、RTL 与参考模型对比都会被记录)\n");
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
