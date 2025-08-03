`timescale 1ns/1ps

module tb_fpmul;

// 参数定义
parameter FP16_WIDTH = 16;
parameter FP32_WIDTH = 32;
parameter TEST_COUNT = 50;

// 输入信号
reg        inst_valid;           // 指令有效信号
reg        src_precision;        // 源寄存器精度：0=16bit，1=32bit
reg        dst_precision;        // 目的寄存器精度：0=16bit，1=32bit
reg [31:0] dvr_fpmul_s0;         // 第一输入寄存器
reg [31:0] dvr_fpmul_s1;         // 第二输入寄存器

// 输出信号
wire [31:0] dr_fpmul_d;          // 输出寄存器

reg clk;

// DPI-C 导入 SoftFloat 函数
import "DPI-C" function shortint unsigned dpi_f16_mul(input shortint unsigned a, input shortint unsigned b);
import "DPI-C" function int unsigned dpi_f32_mul(input int unsigned a, input int unsigned b);
import "DPI-C" function int unsigned dpi_get_inexact_flag();
import "DPI-C" function int unsigned dpi_get_underflow_flag();
import "DPI-C" function int unsigned dpi_get_overflow_flag();
import "DPI-C" function int unsigned dpi_get_infinite_flag();
import "DPI-C" function int unsigned dpi_get_invalid_flag();
import "DPI-C" function int unsigned dpi_get_exception_flags();
import "DPI-C" function void dpi_clear_exception_flags();

// 文件句柄和计数器
integer sim_log;
integer pass_count, fail_count, test_num;

// 测试用例数组
reg [15:0] fp16_test_a [0:TEST_COUNT-1];
reg [15:0] fp16_test_b [0:TEST_COUNT-1];
reg [31:0] fp32_test_a [0:TEST_COUNT-1];
reg [31:0] fp32_test_b [0:TEST_COUNT-1];

// 期望结果
reg [15:0] expected_fp16;
reg [31:0] expected_fp32;

// 实例化被测模块
fpmul uut (
    .inst_valid(inst_valid),
    .src_precision(src_precision),
    .dst_precision(dst_precision),
    .dvr_fpmul_s0(dvr_fpmul_s0),
    .dvr_fpmul_s1(dvr_fpmul_s1),
    .dr_fpmul_d(dr_fpmul_d)
);

// 时钟生成
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// FSDB波形转储
initial begin
    $fsdbDumpfile("tb_fpmul.fsdb");
    $fsdbDumpvars(0, tb_fpmul);
end

// 初始化测试用例
initial begin
    // FP16测试用例初始化
    initialize_fp16_test_cases();
    // FP32测试用例初始化  
    initialize_fp32_test_cases();
end

// FP16测试用例初始化任务
task initialize_fp16_test_cases;
    begin
        // 基本数值测试
        fp16_test_a[0] = 16'h3c00; fp16_test_b[0] = 16'h3c00; // 1.0 * 1.0
        fp16_test_a[1] = 16'h4000; fp16_test_b[1] = 16'h3c00; // 2.0 * 1.0
        fp16_test_a[2] = 16'h3c00; fp16_test_b[2] = 16'h4000; // 1.0 * 2.0
        fp16_test_a[3] = 16'h4000; fp16_test_b[3] = 16'h4000; // 2.0 * 2.0
        fp16_test_a[4] = 16'h3800; fp16_test_b[4] = 16'h3800; // 0.5 * 0.5
        
        // 负数测试
        fp16_test_a[5] = 16'hbc00; fp16_test_b[5] = 16'h3c00; // -1.0 * 1.0
        fp16_test_a[6] = 16'h3c00; fp16_test_b[6] = 16'hbc00; // 1.0 * -1.0
        fp16_test_a[7] = 16'hbc00; fp16_test_b[7] = 16'hbc00; // -1.0 * -1.0
        fp16_test_a[8] = 16'hc000; fp16_test_b[8] = 16'h4000; // -2.0 * 2.0
        
        // 零值测试
        fp16_test_a[9] = 16'h0000; fp16_test_b[9] = 16'h3c00;  // +0 * 1.0
        fp16_test_a[10] = 16'h3c00; fp16_test_b[10] = 16'h0000; // 1.0 * +0
        fp16_test_a[11] = 16'h8000; fp16_test_b[11] = 16'h3c00; // -0 * 1.0
        fp16_test_a[12] = 16'h0000; fp16_test_b[12] = 16'h0000; // +0 * +0
        fp16_test_a[13] = 16'h8000; fp16_test_b[13] = 16'h8000; // -0 * -0
        
        // 无穷大测试
        fp16_test_a[14] = 16'h7c00; fp16_test_b[14] = 16'h3c00; // +Inf * 1.0
        fp16_test_a[15] = 16'h3c00; fp16_test_b[15] = 16'h7c00; // 1.0 * +Inf
        fp16_test_a[16] = 16'hfc00; fp16_test_b[16] = 16'h3c00; // -Inf * 1.0
        fp16_test_a[17] = 16'h7c00; fp16_test_b[17] = 16'h7c00; // +Inf * +Inf
        fp16_test_a[18] = 16'h7c00; fp16_test_b[18] = 16'hfc00; // +Inf * -Inf
        fp16_test_a[19] = 16'h7c00; fp16_test_b[19] = 16'h0000; // +Inf * 0 (NaN)
        
        // NaN测试
        fp16_test_a[20] = 16'h7c01; fp16_test_b[20] = 16'h3c00; // NaN * 1.0
        fp16_test_a[21] = 16'h3c00; fp16_test_b[21] = 16'h7c01; // 1.0 * NaN
        fp16_test_a[22] = 16'h7c01; fp16_test_b[22] = 16'h7c01; // NaN * NaN
        fp16_test_a[23] = 16'h7fff; fp16_test_b[23] = 16'h3c00; // QNaN * 1.0
        
        // 非规格化数测试
        fp16_test_a[24] = 16'h0001; fp16_test_b[24] = 16'h3c00; // 最小非规格化数 * 1.0
        fp16_test_a[25] = 16'h03ff; fp16_test_b[25] = 16'h3c00; // 最大非规格化数 * 1.0
        fp16_test_a[26] = 16'h0001; fp16_test_b[26] = 16'h0001; // 非规格化数 * 非规格化数
        fp16_test_a[27] = 16'h8001; fp16_test_b[27] = 16'h0001; // 负非规格化数测试
        
        // 边界值测试
        fp16_test_a[28] = 16'h7bff; fp16_test_b[28] = 16'h3c00; // 最大规格化数 * 1.0
        fp16_test_a[29] = 16'h0400; fp16_test_b[29] = 16'h3c00; // 最小规格化数 * 1.0
        fp16_test_a[30] = 16'h7bff; fp16_test_b[30] = 16'h7bff; // 最大值相乘（可能溢出）
        fp16_test_a[31] = 16'h0400; fp16_test_b[31] = 16'h0400; // 最小值相乘（可能下溢）
        
        // 特殊数值测试
        fp16_test_a[32] = 16'h4400; fp16_test_b[32] = 16'h3e00; // 4.0 * 1.5
        fp16_test_a[33] = 16'h4800; fp16_test_b[33] = 16'h3400; // 8.0 * 0.25
        fp16_test_a[34] = 16'h5400; fp16_test_b[34] = 16'h2c00; // 64.0 * 0.0625
        fp16_test_a[35] = 16'h3c01; fp16_test_b[35] = 16'h3c01; // (1+ε) * (1+ε)
        
        // 舍入测试用例
        fp16_test_a[36] = 16'h3bff; fp16_test_b[36] = 16'h4000; // (1-ε) * 2
        fp16_test_a[37] = 16'h4001; fp16_test_b[37] = 16'h3fff; // 精度边界测试
        fp16_test_a[38] = 16'h7800; fp16_test_b[38] = 16'h0800; // 大数 * 小数
        fp16_test_a[39] = 16'h0800; fp16_test_b[39] = 16'h7800; // 小数 * 大数
        
        // 指数边界测试
        fp16_test_a[40] = 16'h7800; fp16_test_b[40] = 16'h7800; // 接近溢出
        fp16_test_a[41] = 16'h0200; fp16_test_b[41] = 16'h0200; // 接近下溢
        fp16_test_a[42] = 16'h7a00; fp16_test_b[42] = 16'h0600; // 混合边界
        fp16_test_a[43] = 16'h7000; fp16_test_b[43] = 16'h1000; // 中等指数测试
        
        // 随机测试用例
        fp16_test_a[44] = 16'h5678; fp16_test_b[44] = 16'h1234; // 随机值1
        fp16_test_a[45] = 16'habcd; fp16_test_b[45] = 16'h4321; // 随机值2
        fp16_test_a[46] = 16'h2468; fp16_test_b[46] = 16'h8642; // 随机值3
        fp16_test_a[47] = 16'h1357; fp16_test_b[47] = 16'h9753; // 随机值4
        fp16_test_a[48] = 16'hefef; fp16_test_b[48] = 16'h1010; // 随机值5
        fp16_test_a[49] = 16'h7777; fp16_test_b[49] = 16'h2222; // 随机值6
    end
endtask

// FP32测试用例初始化任务
task initialize_fp32_test_cases;
    begin
        // 基本数值测试
        fp32_test_a[0] = 32'h3f800000; fp32_test_b[0] = 32'h3f800000; // 1.0 * 1.0
        fp32_test_a[1] = 32'h40000000; fp32_test_b[1] = 32'h3f800000; // 2.0 * 1.0
        fp32_test_a[2] = 32'h3f800000; fp32_test_b[2] = 32'h40000000; // 1.0 * 2.0
        fp32_test_a[3] = 32'h40000000; fp32_test_b[3] = 32'h40000000; // 2.0 * 2.0
        fp32_test_a[4] = 32'h3f000000; fp32_test_b[4] = 32'h3f000000; // 0.5 * 0.5
        
        // 负数测试
        fp32_test_a[5] = 32'hbf800000; fp32_test_b[5] = 32'h3f800000; // -1.0 * 1.0
        fp32_test_a[6] = 32'h3f800000; fp32_test_b[6] = 32'hbf800000; // 1.0 * -1.0
        fp32_test_a[7] = 32'hbf800000; fp32_test_b[7] = 32'hbf800000; // -1.0 * -1.0
        fp32_test_a[8] = 32'hc0000000; fp32_test_b[8] = 32'h40000000; // -2.0 * 2.0
        
        // 零值测试
        fp32_test_a[9] = 32'h00000000; fp32_test_b[9] = 32'h3f800000;  // +0 * 1.0
        fp32_test_a[10] = 32'h3f800000; fp32_test_b[10] = 32'h00000000; // 1.0 * +0
        fp32_test_a[11] = 32'h80000000; fp32_test_b[11] = 32'h3f800000; // -0 * 1.0
        fp32_test_a[12] = 32'h00000000; fp32_test_b[12] = 32'h00000000; // +0 * +0
        fp32_test_a[13] = 32'h80000000; fp32_test_b[13] = 32'h80000000; // -0 * -0
        
        // 无穷大测试
        fp32_test_a[14] = 32'h7f800000; fp32_test_b[14] = 32'h3f800000; // +Inf * 1.0
        fp32_test_a[15] = 32'h3f800000; fp32_test_b[15] = 32'h7f800000; // 1.0 * +Inf
        fp32_test_a[16] = 32'hff800000; fp32_test_b[16] = 32'h3f800000; // -Inf * 1.0
        fp32_test_a[17] = 32'h7f800000; fp32_test_b[17] = 32'h7f800000; // +Inf * +Inf
        fp32_test_a[18] = 32'h7f800000; fp32_test_b[18] = 32'hff800000; // +Inf * -Inf
        fp32_test_a[19] = 32'h7f800000; fp32_test_b[19] = 32'h00000000; // +Inf * 0 (NaN)
        
        // NaN测试
        fp32_test_a[20] = 32'h7f800001; fp32_test_b[20] = 32'h3f800000; // NaN * 1.0
        fp32_test_a[21] = 32'h3f800000; fp32_test_b[21] = 32'h7f800001; // 1.0 * NaN
        fp32_test_a[22] = 32'h7f800001; fp32_test_b[22] = 32'h7f800001; // NaN * NaN
        fp32_test_a[23] = 32'h7fffffff; fp32_test_b[23] = 32'h3f800000; // QNaN * 1.0
        
        // 非规格化数测试
        fp32_test_a[24] = 32'h00000001; fp32_test_b[24] = 32'h3f800000; // 最小非规格化数 * 1.0
        fp32_test_a[25] = 32'h007fffff; fp32_test_b[25] = 32'h3f800000; // 最大非规格化数 * 1.0
        fp32_test_a[26] = 32'h00000001; fp32_test_b[26] = 32'h00000001; // 非规格化数 * 非规格化数
        fp32_test_a[27] = 32'h80000001; fp32_test_b[27] = 32'h00000001; // 负非规格化数测试
        
        // 边界值测试
        fp32_test_a[28] = 32'h7f7fffff; fp32_test_b[28] = 32'h3f800000; // 最大规格化数 * 1.0
        fp32_test_a[29] = 32'h00800000; fp32_test_b[29] = 32'h3f800000; // 最小规格化数 * 1.0
        fp32_test_a[30] = 32'h7f7fffff; fp32_test_b[30] = 32'h7f7fffff; // 最大值相乘（溢出）
        fp32_test_a[31] = 32'h00800000; fp32_test_b[31] = 32'h00800000; // 最小值相乘（下溢）
        
        // 特殊数值测试
        fp32_test_a[32] = 32'h40800000; fp32_test_b[32] = 32'h3fc00000; // 4.0 * 1.5
        fp32_test_a[33] = 32'h41000000; fp32_test_b[33] = 32'h3e800000; // 8.0 * 0.25
        fp32_test_a[34] = 32'h42800000; fp32_test_b[34] = 32'h3d800000; // 64.0 * 0.0625
        fp32_test_a[35] = 32'h3f800001; fp32_test_b[35] = 32'h3f800001; // (1+ε) * (1+ε)
        
        // 舍入测试用例
        fp32_test_a[36] = 32'h3f7fffff; fp32_test_b[36] = 32'h40000000; // (1-ε) * 2
        fp32_test_a[37] = 32'h40000001; fp32_test_b[37] = 32'h3fffffff; // 精度边界测试
        fp32_test_a[38] = 32'h7f000000; fp32_test_b[38] = 32'h01000000; // 大数 * 小数
        fp32_test_a[39] = 32'h01000000; fp32_test_b[39] = 32'h7f000000; // 小数 * 大数
        
        // 指数边界测试
        fp32_test_a[40] = 32'h7f000000; fp32_test_b[40] = 32'h7f000000; // 接近溢出
        fp32_test_a[41] = 32'h01000000; fp32_test_b[41] = 32'h01000000; // 接近下溢
        fp32_test_a[42] = 32'h7e000000; fp32_test_b[42] = 32'h02000000; // 混合边界
        fp32_test_a[43] = 32'h60000000; fp32_test_b[43] = 32'h20000000; // 中等指数测试
        
        // 随机测试用例
        fp32_test_a[44] = 32'h56789abc; fp32_test_b[44] = 32'h12345678; // 随机值1
        fp32_test_a[45] = 32'habcdef01; fp32_test_b[45] = 32'h43218765; // 随机值2
        fp32_test_a[46] = 32'h24681357; fp32_test_b[46] = 32'h86420975; // 随机值3
        fp32_test_a[47] = 32'h13579246; fp32_test_b[47] = 32'h97531864; // 随机值4
        fp32_test_a[48] = 32'hefef1010; fp32_test_b[48] = 32'h10101010; // 随机值5
        fp32_test_a[49] = 32'h77777777; fp32_test_b[49] = 32'h22222222; // 随机值6
    end
endtask

// 主测试流程
initial begin
    // 打开日志文件
    sim_log = $fopen("tb_fpmul.log", "w");
    if (sim_log == 0) begin
        $display("错误: 无法打开日志文件");
        $finish;
    end
    
    $fdisplay(sim_log, "FPMUL 测试开始，时间: %t", $time);
    $fdisplay(sim_log, "========================================");
    
    // 初始化计数器
    pass_count = 0;
    fail_count = 0;
    test_num = 0;
    
    // 初始化信号
    inst_valid = 0;
    src_precision = 0;
    dst_precision = 0;
    dvr_fpmul_s0 = 0;
    dvr_fpmul_s1 = 0;
    
    #10;
    
    // 测试FP16乘法
    $fdisplay(sim_log, "\n开始 FP16 乘法测试...");
    $fdisplay(sim_log, "----------------------------------------");
    test_fp16_multiplication();
    
    // 测试FP32乘法
    $fdisplay(sim_log, "\n开始 FP32 乘法测试...");
    $fdisplay(sim_log, "----------------------------------------");
    test_fp32_multiplication();
    
    // 测试指令无效情况
    $fdisplay(sim_log, "\n测试指令无效情况...");
    $fdisplay(sim_log, "----------------------------------------");
    test_invalid_instruction();
    
    // 输出测试结果统计
    print_test_summary();
    
    // 关闭文件
    $fclose(sim_log);
    
    $display("测试完成！详细结果请查看 tb_fpmul.log");
    $finish;
end

// FP16乘法测试任务
task test_fp16_multiplication;
    integer i;
    begin
        src_precision = 0;  // 16bit精度
        dst_precision = 0;  // 16bit精度
        inst_valid = 1;
        
        for (i = 0; i < TEST_COUNT; i = i + 1) begin
            dvr_fpmul_s0 = {16'h0000, fp16_test_a[i]};
            dvr_fpmul_s1 = {16'h0000, fp16_test_b[i]};
            
            // 获取SoftFloat期望结果
            expected_fp16 = dpi_f16_mul(fp16_test_a[i], fp16_test_b[i]);
            
            #10;
            
            // 检查结果
            check_fp16_result(i, fp16_test_a[i], fp16_test_b[i], dr_fpmul_d[15:0], expected_fp16);
            
            test_num = test_num + 1;
        end
    end
endtask

// FP32乘法测试任务
task test_fp32_multiplication;
    integer i;
    begin
        src_precision = 1;  // 32bit精度
        dst_precision = 1;  // 32bit精度
        inst_valid = 1;
        
        for (i = 0; i < TEST_COUNT; i = i + 1) begin
            dvr_fpmul_s0 = fp32_test_a[i];
            dvr_fpmul_s1 = fp32_test_b[i];
            
            // 获取SoftFloat期望结果
            expected_fp32 = dpi_f32_mul(fp32_test_a[i], fp32_test_b[i]);
            
            #10;
            
            // 检查结果
            check_fp32_result(i, fp32_test_a[i], fp32_test_b[i], dr_fpmul_d, expected_fp32);
            
            test_num = test_num + 1;
        end
    end
endtask

// 测试指令无效情况
task test_invalid_instruction;
    begin
        // 设置测试数据
        dvr_fpmul_s0 = 32'h3f800000; // 1.0
        dvr_fpmul_s1 = 32'h40000000; // 2.0
        src_precision = 1;
        dst_precision = 1;
        
        // 指令无效
        inst_valid = 0;
        #10;
        
        if (dr_fpmul_d !== 32'h00000000) begin
            $fdisplay(sim_log, "错误: 指令无效时输出应为0，实际输出: %h", dr_fpmul_d);
            fail_count = fail_count + 1;
        end else begin
            $fdisplay(sim_log, "通过: 指令无效测试");
            pass_count = pass_count + 1;
        end
        
        test_num = test_num + 1;
    end
endtask

// 检查FP16结果
task check_fp16_result;
    input integer test_index;
    input [15:0] input_a;
    input [15:0] input_b;
    input [15:0] actual_result;
    input [15:0] expected_result;
    
    reg is_actual_nan, is_expected_nan;
    reg match_found, is_inexact;
    reg [15:0] expected_plus_one, expected_minus_one;
    integer exception_flags;
    string flag_info;
    begin
        // 获取异常标志
        exception_flags = dpi_get_exception_flags();
        is_inexact = (exception_flags & dpi_get_inexact_flag()) != 0;
        
        // 解析异常标志
        flag_info = "";
        if (exception_flags & dpi_get_inexact_flag()) flag_info = {flag_info, " 不精确"};
        if (exception_flags & dpi_get_underflow_flag()) flag_info = {flag_info, " 下溢"};
        if (exception_flags & dpi_get_overflow_flag()) flag_info = {flag_info, " 上溢"};
        if (exception_flags & dpi_get_infinite_flag()) flag_info = {flag_info, " 无穷大"};
        if (exception_flags & dpi_get_invalid_flag()) flag_info = {flag_info, " 无效"};
        if (flag_info == "") flag_info = " 无异常";
        
        // 检查NaN情况
        is_actual_nan = (actual_result[14:10] == 5'b11111) && (actual_result[9:0] != 10'b0);
        is_expected_nan = (expected_result[14:10] == 5'b11111) && (expected_result[9:0] != 10'b0);
        
        match_found = 0;
        
        if (is_expected_nan && is_actual_nan) begin
            match_found = 1;
        end else if (actual_result === expected_result) begin
            match_found = 1;
        end else if (is_inexact) begin
            // 当结果不精确时，允许最低位±1的误差
            expected_plus_one = expected_result + 1;
            expected_minus_one = expected_result - 1;
            if ((actual_result === expected_plus_one) || (actual_result === expected_minus_one)) begin
                match_found = 1;
            end
        end
        
        if (match_found) begin
            if (is_expected_nan && is_actual_nan) begin
                $fdisplay(sim_log, "FP16 测试 %0d: 通过 (NaN) - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                         test_index, input_a, input_b, expected_result, actual_result, flag_info);
            end else if (actual_result === expected_result) begin
                $fdisplay(sim_log, "FP16 测试 %0d: 通过 - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                         test_index, input_a, input_b, expected_result, actual_result, flag_info);
            end else begin
                $fdisplay(sim_log, "FP16 测试 %0d: 通过 (±1 tolerance) - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                         test_index, input_a, input_b, expected_result, actual_result, flag_info);
            end
            pass_count = pass_count + 1;
        end else begin
            $fdisplay(sim_log, "FP16 测试 %0d: 失败 - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                     test_index, input_a, input_b, expected_result, actual_result, flag_info);
            fail_count = fail_count + 1;
        end
        
        // 清除异常标志
        dpi_clear_exception_flags();
    end
endtask

// 检查FP32结果
task check_fp32_result;
    input integer test_index;
    input [31:0] input_a;
    input [31:0] input_b;
    input [31:0] actual_result;
    input [31:0] expected_result;
    
    reg is_actual_nan, is_expected_nan;
    reg match_found, is_inexact;
    reg [31:0] expected_plus_one, expected_minus_one;
    integer exception_flags;
    string flag_info;
    begin
        // 获取异常标志
        exception_flags = dpi_get_exception_flags();
        is_inexact = (exception_flags & dpi_get_inexact_flag()) != 0;
        
        // 解析异常标志
        flag_info = "";
        if (exception_flags & dpi_get_inexact_flag()) flag_info = {flag_info, " 不精确"};
        if (exception_flags & dpi_get_underflow_flag()) flag_info = {flag_info, " 下溢"};
        if (exception_flags & dpi_get_overflow_flag()) flag_info = {flag_info, " 上溢"};
        if (exception_flags & dpi_get_infinite_flag()) flag_info = {flag_info, " 无穷大"};
        if (exception_flags & dpi_get_invalid_flag()) flag_info = {flag_info, " 无效"};
        if (flag_info == "") flag_info = " 无异常";
        
        // 检查NaN情况
        is_actual_nan = (actual_result[30:23] == 8'b11111111) && (actual_result[22:0] != 23'b0);
        is_expected_nan = (expected_result[30:23] == 8'b11111111) && (expected_result[22:0] != 23'b0);
        
        match_found = 0;
        
        if (is_expected_nan && is_actual_nan) begin
            match_found = 1;
        end else if (actual_result === expected_result) begin
            match_found = 1;
        end else if (is_inexact) begin
            // 当结果不精确时，允许最低位±1的误差
            expected_plus_one = expected_result + 1;
            expected_minus_one = expected_result - 1;
            if ((actual_result === expected_plus_one) || (actual_result === expected_minus_one)) begin
                match_found = 1;
            end
        end
        
        if (match_found) begin
            if (is_expected_nan && is_actual_nan) begin
                $fdisplay(sim_log, "FP32 测试 %0d: 通过 (NaN) - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                         test_index, input_a, input_b, expected_result, actual_result, flag_info);
            end else if (actual_result === expected_result) begin
                $fdisplay(sim_log, "FP32 测试 %0d: 通过 - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                         test_index, input_a, input_b, expected_result, actual_result, flag_info);
            end else begin
                $fdisplay(sim_log, "FP32 测试 %0d: 通过 (±1 tolerance) - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                         test_index, input_a, input_b, expected_result, actual_result, flag_info);
            end
            pass_count = pass_count + 1;
        end else begin
            $fdisplay(sim_log, "FP32 测试 %0d: 失败 - A=%h, B=%h, 期望=%h, 实际=%h | 异常标志:%s", 
                     test_index, input_a, input_b, expected_result, actual_result, flag_info);
            fail_count = fail_count + 1;
        end
        
        // 清除异常标志
        dpi_clear_exception_flags();
    end
endtask

// 打印测试结果统计
task print_test_summary;
    begin
        $fdisplay(sim_log, "\n========================================");
        $fdisplay(sim_log, "测试结果统计:");
        $fdisplay(sim_log, "========================================");
        $fdisplay(sim_log, "总测试数: %0d", test_num);
        $fdisplay(sim_log, "通过数: %0d", pass_count);
        $fdisplay(sim_log, "失败数: %0d", fail_count);
        $fdisplay(sim_log, "通过率: %0.2f%%", (pass_count * 100.0) / test_num);
        
        if (fail_count == 0) begin
            $fdisplay(sim_log, "\n🎉 所有测试都通过了！");
            $display("✅ 所有测试都通过了！");
        end else begin
            $fdisplay(sim_log, "\n❌ 有 %0d 个测试失败，请检查错误日志", fail_count);
            $display("❌ 有 %0d 个测试失败，请检查 tb_fpmul_errors.log", fail_count);
        end
        
        $fdisplay(sim_log, "测试结束时间: %t", $time);
    end
endtask

endmodule
