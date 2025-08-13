`timescale 1ns / 1ps

module intadd_tb;

    // DPI-C接口声明，匹配intadd_interface.c
    import "DPI-C" function void add32_128bit(
        input longint unsigned src0_high, input longint unsigned src0_low,
        input longint unsigned src1_high, input longint unsigned src1_low,
        input int sign_s0, input int sign_s1,
        output longint unsigned dst_high, output longint unsigned dst_low
    );
    import "DPI-C" function void add8_128bit(
        input longint unsigned src0_high, input longint unsigned src0_low,
        input longint unsigned src1_high, input longint unsigned src1_low,
        input longint unsigned src2_high, input longint unsigned src2_low,
        input int sign_s0, input int sign_s1, input int sign_s2,
        output longint unsigned dst0_high, output longint unsigned dst0_low,
        output longint unsigned dst1_high, output longint unsigned dst1_low
    );

    reg [127:0] src_reg0, src_reg1, src_reg2;
    reg [1:0]   precision_s0, precision_s1, precision_s2;
    reg         sign_s0, sign_s1, sign_s2;
    reg         inst_valid;

    wire [127:0] dst_reg0, dst_reg1;

    // UUT
    intadd uut (
        .src_reg0(src_reg0),
        .src_reg1(src_reg1),
        .src_reg2(src_reg2),
        .precision_s0(precision_s0),
        .precision_s1(precision_s1),
        .precision_s2(precision_s2),
        .sign_s0(sign_s0),
        .sign_s1(sign_s1),
        .sign_s2(sign_s2),
        .inst_valid(inst_valid),
        .dst_reg0(dst_reg0),
        .dst_reg1(dst_reg1)
    );

    integer result_file;
    longint unsigned c_dst0_high, c_dst0_low, c_dst1_high, c_dst1_low;

    // 初始化输入
    task init_inputs;
    begin
        src_reg0 = 0; src_reg1 = 0; src_reg2 = 0;
        precision_s0 = 0; precision_s1 = 0; precision_s2 = 0;
        sign_s0 = 0; sign_s1 = 0; sign_s2 = 0;
        inst_valid = 0;
    end
    endtask

    // 32bit模式对比
    function automatic int compare_and_log_32bit(
        input [127:0] hw_dst0,
        input [127:0] src0, src1,
        input         sign0, sign1
    );
    begin
        add32_128bit(src0[127:64], src0[63:0], src1[127:64], src1[63:0], sign0, sign1, c_dst0_high, c_dst0_low);
        result_file = $fopen("result.txt", "a");
        $fdisplay(result_file, "==== 32bit mode ====");
        $fdisplay(result_file, "src0=0x%h", src0);
        $fdisplay(result_file, "src1=0x%h", src1);
        $fdisplay(result_file, "sign0=%0d, sign1=%0d", sign0, sign1);
        $fdisplay(result_file, "HW: dst0=0x%h", hw_dst0);
        $fdisplay(result_file, "C : dst0=0x%016h%016h", c_dst0_high, c_dst0_low);
        if (hw_dst0 !== {c_dst0_high, c_dst0_low}) begin
            $fdisplay(result_file, "RESULT: ERROR");
            compare_and_log_32bit = 0;
        end else begin
            $fdisplay(result_file, "RESULT: PASS");
            compare_and_log_32bit = 1;
        end
        $fdisplay(result_file, "==== END ====\n");
        $fclose(result_file);
    end
    endfunction

    // 4+8bit模式对比
    function automatic int compare_and_log_4_8bit(
        input [127:0] hw_dst0, hw_dst1,
        input [127:0] src0, src1, src2,
        input         sign0, sign1, sign2
    );
    begin
        add8_128bit(src0[127:64], src0[63:0], src1[127:64], src1[63:0], src2[127:64], src2[63:0], sign0, sign1, sign2, c_dst0_high, c_dst0_low, c_dst1_high, c_dst1_low);
        result_file = $fopen("result.txt", "a");
        $fdisplay(result_file, "==== 4+8bit mode ====");
        $fdisplay(result_file, "src0=0x%h", src0);
        $fdisplay(result_file, "src1=0x%h", src1);
        $fdisplay(result_file, "src2=0x%h", src2);
        $fdisplay(result_file, "sign0=%0d, sign1=%0d, sign2=%0d", sign0, sign1, sign2);
        $fdisplay(result_file, "HW: dst0=0x%h, dst1=0x%h", hw_dst0, hw_dst1);
        $fdisplay(result_file, "C : dst0=0x%016h%016h, dst1=0x%016h%016h", c_dst0_high, c_dst0_low, c_dst1_high, c_dst1_low);
        if (hw_dst0 !== {c_dst0_high, c_dst0_low} || hw_dst1 !== {c_dst1_high, c_dst1_low}) begin
            $fdisplay(result_file, "RESULT: ERROR");
            compare_and_log_4_8bit = 0;
        end else begin
            $fdisplay(result_file, "RESULT: PASS");
            compare_and_log_4_8bit = 1;
        end
        $fdisplay(result_file, "==== END ====\n");
        $fclose(result_file);
    end
    endfunction

    integer i;
    integer pass_cnt, total_cnt;

    initial begin
        pass_cnt = 0;
        total_cnt = 0;
        result_file = $fopen("result.txt", "w");
        $fdisplay(result_file, "==== INTADD TB TEST START ====");
        $fclose(result_file);

        init_inputs;
        #10;

        // 32bit模式测试
        for (i = 0; i < 10000; i = i + 1) begin
            src_reg0 = {$random, $random, $random, $random};
            src_reg1 = {$random, $random, $random, $random};
            precision_s0 = 2'b11;
            precision_s1 = 2'b11;
            sign_s0 = $random & 1;
            sign_s1 = $random & 1;
            inst_valid = 1;
            #5;
            if (compare_and_log_32bit(dst_reg0, src_reg0, src_reg1, sign_s0, sign_s1))
                pass_cnt = pass_cnt + 1;
            total_cnt = total_cnt + 1;
            #5;
        end

        // 4+8bit模式测试
        for (i = 0; i < 10000; i = i + 1) begin
            src_reg0 = {$random, $random, $random, $random};
            src_reg1 = {$random, $random, $random, $random};
            src_reg2 = {$random, $random, $random, $random};
            precision_s0 = 2'b00;
            precision_s1 = 2'b00;
            precision_s2 = 2'b00;
            sign_s0 = $random & 1;
            sign_s1 = $random & 1;
            sign_s2 = $random & 1;
            inst_valid = 1;
            #5;
            if (compare_and_log_4_8bit(dst_reg0, dst_reg1, src_reg0, src_reg1, src_reg2, sign_s0, sign_s1, sign_s2))
                pass_cnt = pass_cnt + 1;
            total_cnt = total_cnt + 1;
            #5;
        end

        // 写入统计结果
        result_file = $fopen("result.txt", "a");
        $fdisplay(result_file, "==== SUMMARY ====");
        $fdisplay(result_file, "PASS: %0d / TOTAL: %0d", pass_cnt, total_cnt);
        $fdisplay(result_file, "==== END ====");
        $fclose(result_file);

        $display("All tests completed.");
        $finish;
    end

endmodule