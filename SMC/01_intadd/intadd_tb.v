`timescale 1ns / 1ps

module intadd_tb;

    // DPI-C接口声明
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
    reg         inst_valid;
    reg         clk;
    reg         rst_n;

    wire [127:0] dst_reg0, dst_reg1, st;

    // 时钟生成
    always #5 clk = ~clk;

    // UUT
    intadd uut (
        .clk(clk),
        .rst_n(rst_n),
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
        .dst_reg1(dst_reg1),
        .st(st)
    );

    integer result_file;
    longint unsigned c_dst0_high, c_dst0_low, c_dst1_high, c_dst1_low;
    longint unsigned c_st_high, c_st_low;

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
        input [127:0] hw_st,
        input [127:0] src0, src1,
        input         sign0, sign1
    );
    begin
        add32_128bit(src0[127:64], src0[63:0], 
                     src1[127:64], src1[63:0], 
                     sign0, sign1, 
                     c_dst0_high, c_dst0_low,
                     c_st_high, c_st_low);
                     
        result_file = $fopen("result.txt", "a");
        $fdisplay(result_file, "==== 32bit mode ====");
        $fdisplay(result_file, "src0=0x%h", src0);
        $fdisplay(result_file, "src1=0x%h", src1);
        $fdisplay(result_file, "sign0=%0d, sign1=%0d", sign0, sign1);
        $fdisplay(result_file, "HW: dst0=0x%h, st=0x%h", hw_dst0, hw_st);
        $fdisplay(result_file, "C : dst0=0x%016h%016h, st=0x%016h%016h", 
                  c_dst0_high, c_dst0_low, c_st_high, c_st_low);
                  
        // 状态寄存器特殊验证
        if (hw_st !== {c_st_high, c_st_low}) begin
            $fdisplay(result_file, "STATUS ERROR: HW st != C st");
            $fdisplay(result_file, "  HW st bits:");
            $fdisplay(result_file, "    [2:0]   = %b", hw_st[2:0]);
            $fdisplay(result_file, "    [34:32] = %b", hw_st[34:32]);
            $fdisplay(result_file, "    [66:64] = %b", hw_st[66:64]);
            $fdisplay(result_file, "    [98:96] = %b", hw_st[98:96]);
            $fdisplay(result_file, "  C st bits:");
            $fdisplay(result_file, "    [2:0]   = %b", {c_st_low[2], c_st_low[1], c_st_low[0]});
            $fdisplay(result_file, "    [34:32] = %b", {c_st_low[34], c_st_low[33], c_st_low[32]});
            $fdisplay(result_file, "    [66:64] = %b", {c_st_high[2], c_st_high[1], c_st_high[0]});
            $fdisplay(result_file, "    [98:96] = %b", {c_st_high[34], c_st_high[33], c_st_high[32]});
        end
                  
        if (hw_dst0 !== {c_dst0_high, c_dst0_low} || 
            hw_st !== {c_st_high, c_st_low}) begin
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
        input [127:0] hw_st,
        input [127:0] src0, src1, src2,
        input         sign0, sign1, sign2
    );
    begin
        add8_128bit(src0[127:64], src0[63:0], 
                    src1[127:64], src1[63:0], 
                    src2[127:64], src2[63:0], 
                    sign0, sign1, sign2, 
                    c_dst0_high, c_dst0_low, 
                    c_dst1_high, c_dst1_low,
                    c_st_high, c_st_low);
                    
        result_file = $fopen("result.txt", "a");
        $fdisplay(result_file, "==== 4+8bit mode ====");
        $fdisplay(result_file, "src0=0x%h", src0);
        $fdisplay(result_file, "src1=0x%h", src1);
        $fdisplay(result_file, "src2=0x%h", src2);
        $fdisplay(result_file, "sign0=%0d, sign1=%0d, sign2=%0d", sign0, sign1, sign2);
        $fdisplay(result_file, "HW: dst0=0x%h, dst1=0x%h, st=0x%h", hw_dst0, hw_dst1, hw_st);
        $fdisplay(result_file, "C : dst0=0x%016h%016h, dst1=0x%016h%016h, st=0x%016h%016h", 
                  c_dst0_high, c_dst0_low, c_dst1_high, c_dst1_low, c_st_high, c_st_low);
                  
        // 状态寄存器特殊验证
        if (hw_st !== 0) begin
            $fdisplay(result_file, "STATUS ERROR: HW st should be 0 in 4+8bit mode");
        end
                  
        if (hw_dst0 !== {c_dst0_high, c_dst0_low} || 
            hw_dst1 !== {c_dst1_high, c_dst1_low} || 
            hw_st !== 0) begin
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
        // 初始化时钟和复位
        clk = 0;
        rst_n = 0;
        pass_cnt = 0;
        total_cnt = 0;
        
        result_file = $fopen("result.txt", "w");
        $fdisplay(result_file, "==== INTADD TB TEST START ====");
        $fclose(result_file);

        init_inputs;
        #10 rst_n = 1;  // 释放复位

        // 边界测试用例
        // 测试1: 有符号正溢出
        $display("Running boundary test: Signed positive overflow");
        src_reg0 = {32'h7FFFFFFF, 32'h7FFFFFFF, 32'h7FFFFFFF, 32'h7FFFFFFF};
        src_reg1 = {32'h00000001, 32'h00000001, 32'h00000001, 32'h00000001};
        precision_s0 = 2'b11;
        precision_s1 = 2'b11;
        sign_s0 = 1;
        sign_s1 = 1;
        inst_valid = 1;
        #10;
        if (compare_and_log_32bit(dst_reg0, st, src_reg0, src_reg1, sign_s0, sign_s1))
            pass_cnt = pass_cnt + 1;
        total_cnt = total_cnt + 1;
        #5;
        
        // 测试2: 有符号负溢出
        $display("Running boundary test: Signed negative overflow");
        src_reg0 = {32'h80000000, 32'h80000000, 32'h80000000, 32'h80000000};
        src_reg1 = {32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF}; // -1
        precision_s0 = 2'b11;
        precision_s1 = 2'b11;
        sign_s0 = 1;
        sign_s1 = 1;
        inst_valid = 1;
        #10;
        if (compare_and_log_32bit(dst_reg0, st, src_reg0, src_reg1, sign_s0, sign_s1))
            pass_cnt = pass_cnt + 1;
        total_cnt = total_cnt + 1;
        #5;
        
        // 测试3: 无符号溢出
        $display("Running boundary test: Unsigned overflow");
        src_reg0 = {32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF};
        src_reg1 = {32'h00000001, 32'h00000001, 32'h00000001, 32'h00000001};
        precision_s0 = 2'b11;
        precision_s1 = 2'b11;
        sign_s0 = 0;
        sign_s1 = 0;
        inst_valid = 1;
        #10;
        if (compare_and_log_32bit(dst_reg0, st, src_reg0, src_reg1, sign_s0, sign_s1))
            pass_cnt = pass_cnt + 1;
        total_cnt = total_cnt + 1;
        #5;
        
        // 测试4: 混合符号比较
        $display("Running boundary test: Mixed sign comparison");
        src_reg0 = {32'hFFFFFFFF, 32'h7FFFFFFF, 32'h00000000, 32'h80000000}; // -1, MAX, 0, MIN
        src_reg1 = {32'h00000001, 32'h80000000, 32'hFFFFFFFF, 32'h7FFFFFFF}; // 1, MIN, -1, MAX
        precision_s0 = 2'b11;
        precision_s1 = 2'b11;
        sign_s0 = 1;
        sign_s1 = 1;
        inst_valid = 1;
        #10;
        if (compare_and_log_32bit(dst_reg0, st, src_reg0, src_reg1, sign_s0, sign_s1))
            pass_cnt = pass_cnt + 1;
        total_cnt = total_cnt + 1;
        #5;
        
        // 随机测试
        $display("Starting random tests...");
        
        // 32bit模式测试
        for (i = 0; i < 1000; i = i + 1) begin
            src_reg0 = {$random, $random, $random, $random};
            src_reg1 = {$random, $random, $random, $random};
            precision_s0 = 2'b11;
            precision_s1 = 2'b11;
            sign_s0 = $random & 1;
            sign_s1 = $random & 1;
            inst_valid = 1;
            #10;
            if (compare_and_log_32bit(dst_reg0, st, src_reg0, src_reg1, sign_s0, sign_s1))
                pass_cnt = pass_cnt + 1;
            total_cnt = total_cnt + 1;
            #5;
        end

        // 4+8bit模式测试
        for (i = 0; i < 1000; i = i + 1) begin
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
            #10;
            if (compare_and_log_4_8bit(dst_reg0, dst_reg1, st, src_reg0, src_reg1, src_reg2, sign_s0, sign_s1, sign_s2))
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

        $display("All tests completed. PASS: %0d / TOTAL: %0d", pass_cnt, total_cnt);
        $finish;
    end

endmodule