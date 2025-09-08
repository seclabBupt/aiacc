`timescale 1ns/1ps

module tb_fp16_to_fp32_multiplier;

  // SoftFloat舍入模式
  localparam SOFTFLOAT_ROUND_NEAR_EVEN   = 0;  // 四舍五入到最近的偶数
  localparam SOFTFLOAT_ROUND_MINMAG      = 1;  // 向零舍入
  localparam SOFTFLOAT_ROUND_MIN         = 2;  // 向负无穷大舍入
  localparam SOFTFLOAT_ROUND_MAX         = 3;  // 向正无穷大舍入
  localparam SOFTFLOAT_ROUND_NEAR_MAXMAG = 4;  // 四舍五入到最近，关系到最大幅度
  
  // 设计参数
  localparam NUM_GROUPS = 32;          // 组数
  localparam NUM_S0123_REGS = 2;       // S0-S3寄存器数量
  localparam NUM_S4567_REGS = 2;       // S4-S7寄存器数量
  
  // ==========================================================================
  // DUT接口信号
  // ==========================================================================
  reg clk;
  reg rst_n;
  
  // 输入寄存器
  reg [127:0] dvr_fp16mul_s0;     // 第一个fp16的低4位
  reg [127:0] dvr_fp16mul_s1;     // 第一个fp16的次低4位
  reg [127:0] dvr_fp16mul_s2;     // 第一个fp16的次高4位
  reg [127:0] dvr_fp16mul_s3;     // 第一个fp16的高4位
  reg [127:0] dvr_fp16mul_s4567;  // 第二个fp16的4个4位段
  
  // 控制信号
  reg [1:0] cru_fp16mul_s0123;    // s0-s3控制
  reg [3:0] cru_fp16mul_s4567;    // s4567控制
  reg [2:0] cru_fp16mul;          // 主控制信号
  
  // 输出寄存器
  wire [127:0] dr_fp16mul_d0;     // fp32结果的位[3:0]
  wire [127:0] dr_fp16mul_d1;     // fp32结果的位[7:4]
  wire [127:0] dr_fp16mul_d2;     // fp32结果的位[11:8]
  wire [127:0] dr_fp16mul_d3;     // fp32结果的位[15:12]
  wire [127:0] dr_fp16mul_d4;     // fp32结果的位[19:16]
  wire [127:0] dr_fp16mul_d5;     // fp32结果的位[23:20]
  wire [127:0] dr_fp16mul_d6;     // fp32结果的位[27:24]
  wire [127:0] dr_fp16mul_d7;     // fp32结果的位[31:28]
  
  import "DPI-C" function int unsigned fp16_inputs_mul_to_fp32_softfloat(
    input shortint unsigned a, input shortint unsigned b
  );
  import "DPI-C" function void set_softfloat_rounding_mode(input int unsigned mode);
  import "DPI-C" function void clear_softfloat_flags();
  import "DPI-C" function int unsigned get_softfloat_flags();
  
  integer test_count = 0;
  integer pass_count = 0;
  integer fail_count = 0;
  integer expected_fp32;
  integer softfloat_flags;
  integer random_num = 1000;
  
  fp16_to_fp32_multiplier uut (
    .clk(clk),
    .rst_n(rst_n),
    .dvr_fp16mul_s0(dvr_fp16mul_s0),
    .dvr_fp16mul_s1(dvr_fp16mul_s1),
    .dvr_fp16mul_s2(dvr_fp16mul_s2),
    .dvr_fp16mul_s3(dvr_fp16mul_s3),
    .dvr_fp16mul_s4567(dvr_fp16mul_s4567),
    .cru_fp16mul_s0123(cru_fp16mul_s0123),
    .cru_fp16mul_s4567(cru_fp16mul_s4567),
    .cru_fp16mul(cru_fp16mul),
    .dr_fp16mul_d0(dr_fp16mul_d0),
    .dr_fp16mul_d1(dr_fp16mul_d1),
    .dr_fp16mul_d2(dr_fp16mul_d2),
    .dr_fp16mul_d3(dr_fp16mul_d3),
    .dr_fp16mul_d4(dr_fp16mul_d4),
    .dr_fp16mul_d5(dr_fp16mul_d5),
    .dr_fp16mul_d6(dr_fp16mul_d6),
    .dr_fp16mul_d7(dr_fp16mul_d7)
  );
  
  initial begin : clock_gen
    clk = 0;
    forever #5 clk = ~clk;  // 10ns周期，100MHz
  end
  
  // VCD波形文件生成
  initial begin
    $dumpfile("fp16_to_fp32_multiplier.vcd");
    $dumpvars(0, tb_fp16_to_fp32_multiplier);
  end
  

  // 复位系统
  task reset_system;
  begin
    rst_n = 0;
    @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  end
  endtask
  

  // 结果验证函数
  function verify_result;
    input [15:0] fp16_a;
    input [15:0] fp16_b;
    input [31:0] actual_result;
    input [31:0] expected_result; 
    input integer test_id;
    input string test_type;
    
    reg is_nan_expected, is_nan_actual;
  begin
    test_count = test_count + 1;
    
    // 特殊处理NaN
    is_nan_expected = (expected_result[30:23] == 8'hff) && (expected_result[22:0] != 0);
    is_nan_actual = (actual_result[30:23] == 8'hff) && (actual_result[22:0] != 0);
    
    if (actual_result == expected_result || (is_nan_expected && is_nan_actual)) begin
      $display("%s测试 %0d: 通过 - a=%h, b=%h, softfloat=%h (标志=%h), 实际值=%h", 
               test_type, test_id, fp16_a, fp16_b, expected_result, get_softfloat_flags(), actual_result);
      verify_result = 1; // 通过
    end else begin
      $display("%s测试 %0d: 失败 - a=%h, b=%h, softfloat=%h (标志=%h), 实际值=%h", 
               test_type, test_id, fp16_a, fp16_b, expected_result, get_softfloat_flags(), actual_result);
      verify_result = 0; // 失败
    end
  end
  endfunction
  
  
  // 测试无效指令
  task test_idle_instruction;
    input [15:0] fp16_a;
    input [15:0] fp16_b;
    input integer group_id;
    input integer reg_bank;
    
    reg [31:0] original_result;
  begin
    $display("开始无效指令测试 (组%0d, 寄存器组%0d)...", group_id, reg_bank);
    
    // 先执行一次正常操作获取结果
    write_fp16_to_registers(fp16_a, fp16_b, group_id, reg_bank, 1'b1);
    original_result = reconstruct_result(group_id);
    
    // 使用无效指令尝试写入
    write_fp16_to_registers(fp16_a, fp16_b, group_id, reg_bank, 1'b0);
    
    // 检查结果是否保持不变
    if (reconstruct_result(group_id) === original_result) begin
      $display("无效指令测试通过: 结果保持不变");
      pass_count = pass_count + 1;
    end else begin
      $display("无效指令测试失败: 结果被改变");
      fail_count = fail_count + 1;
    end
    
    test_count = test_count + 1;
  end
  endtask
  
  // 测试所有寄存器组选择组合
  task test_all_bank_combinations;
    input [15:0] a0, b0; // Bank 0 data
    input [15:0] a1, b1; // Bank 1 data
    input integer group_id;

    reg [31:0] actual_result;
    reg [31:0] expected_result;
    string test_name;
  begin
    $display("开始所有寄存器组组合测试 (组%0d)...", group_id);
    
    // 写入数据到两个寄存器组
    write_s0123_registers(a0, group_id, 0, 1'b1);
    write_s4567_registers(b0, group_id, 0, 1'b1);
    write_s0123_registers(a1, group_id, 1, 1'b1);
    write_s4567_registers(b1, group_id, 1, 1'b1);

    // --- 测试组合 0 * 0 ---
    test_name = "交叉寄存器(0*0)";
    clear_softfloat_flags();
    expected_result = fp16_inputs_mul_to_fp32_softfloat(a0, b0);
    cru_fp16mul = {1'b1, 1'b0, 1'b0}; @(posedge clk); cru_fp16mul = 3'b000; @(posedge clk); repeat(2) @(posedge clk);
    actual_result = reconstruct_result(group_id);
    if (verify_result(a0, b0, actual_result, expected_result, test_count, test_name)) pass_count = pass_count + 1; else fail_count = fail_count + 1;

    // --- 测试组合 0 * 1 ---
    test_name = "交叉寄存器(0*1)";
    clear_softfloat_flags();
    expected_result = fp16_inputs_mul_to_fp32_softfloat(a0, b1);
    cru_fp16mul = {1'b1, 1'b0, 1'b1}; @(posedge clk); cru_fp16mul = 3'b000; @(posedge clk); repeat(2) @(posedge clk);
    actual_result = reconstruct_result(group_id);
    if (verify_result(a0, b1, actual_result, expected_result, test_count, test_name)) pass_count = pass_count + 1; else fail_count = fail_count + 1;

    // --- 测试组合 1 * 0 ---
    test_name = "交叉寄存器(1*0)";
    clear_softfloat_flags();
    expected_result = fp16_inputs_mul_to_fp32_softfloat(a1, b0);
    cru_fp16mul = {1'b1, 1'b1, 1'b0}; @(posedge clk); cru_fp16mul = 3'b000; @(posedge clk); repeat(2) @(posedge clk);
    actual_result = reconstruct_result(group_id);
    if (verify_result(a1, b0, actual_result, expected_result, test_count, test_name)) pass_count = pass_count + 1; else fail_count = fail_count + 1;

    // --- 测试组合 1 * 1 ---
    test_name = "交叉寄存器(1*1)";
    clear_softfloat_flags();
    expected_result = fp16_inputs_mul_to_fp32_softfloat(a1, b1);
    cru_fp16mul = {1'b1, 1'b1, 1'b1}; @(posedge clk); cru_fp16mul = 3'b000; @(posedge clk); repeat(2) @(posedge clk);
    actual_result = reconstruct_result(group_id);
    if (verify_result(a1, b1, actual_result, expected_result, test_count, test_name)) pass_count = pass_count + 1; else fail_count = fail_count + 1;
  end
  endtask


  // 仅写入s0-s3寄存器
  task write_s0123_registers;
    input [15:0] fp16_val;
    input integer group_id;
    input integer reg_bank;
    input instruction_valid;

    reg [127:0] test_s0, test_s1, test_s2, test_s3;
  begin
    test_s0 = 128'h0; test_s1 = 128'h0; test_s2 = 128'h0; test_s3 = 128'h0;
    test_s0[group_id*4 +: 4] = fp16_val[3:0];    
    test_s1[group_id*4 +: 4] = fp16_val[7:4];    
    test_s2[group_id*4 +: 4] = fp16_val[11:8];   
    test_s3[group_id*4 +: 4] = fp16_val[15:12];

    dvr_fp16mul_s0 = test_s0;
    dvr_fp16mul_s1 = test_s1;
    dvr_fp16mul_s2 = test_s2;
    dvr_fp16mul_s3 = test_s3;
    cru_fp16mul_s0123 = {instruction_valid, reg_bank[0]};
    @(posedge clk);
    cru_fp16mul_s0123 = 2'b00;
    @(posedge clk);
  end
  endtask

  // 仅写入s4-s7寄存器
  task write_s4567_registers;
    input [15:0] fp16_val;
    input integer group_id;
    input integer reg_bank;
    input instruction_valid;

    reg [127:0] test_s4, test_s5, test_s6, test_s7;
  begin
    test_s4 = 128'h0; test_s5 = 128'h0; test_s6 = 128'h0; test_s7 = 128'h0;
    test_s4[group_id*4 +: 4] = fp16_val[3:0];    
    test_s5[group_id*4 +: 4] = fp16_val[7:4];    
    test_s6[group_id*4 +: 4] = fp16_val[11:8];   
    test_s7[group_id*4 +: 4] = fp16_val[15:12]; 

    dvr_fp16mul_s4567 = test_s4; cru_fp16mul_s4567 = {instruction_valid, reg_bank[0], 2'b00}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
    dvr_fp16mul_s4567 = test_s5; cru_fp16mul_s4567 = {instruction_valid, reg_bank[0], 2'b01}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
    dvr_fp16mul_s4567 = test_s6; cru_fp16mul_s4567 = {instruction_valid, reg_bank[0], 2'b10}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
    dvr_fp16mul_s4567 = test_s7; cru_fp16mul_s4567 = {instruction_valid, reg_bank[0], 2'b11}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
  end
  endtask
  
  // 写入两个FP16到寄存器
  task write_fp16_to_registers;
    input [15:0] fp16_a;
    input [15:0] fp16_b;
    input integer group_id;
    input integer reg_bank;
    input instruction_valid;
  begin
    write_s0123_registers(fp16_a, group_id, reg_bank, instruction_valid);
    write_s4567_registers(fp16_b, group_id, reg_bank, instruction_valid);
    
    // 执行乘法运算
    if (instruction_valid) begin
      cru_fp16mul = {1'b1, reg_bank[0], reg_bank[0]};  // 有效信号 + 源寄存器选择
      @(posedge clk);
      cru_fp16mul = 3'b000;  // 清除控制信号
      @(posedge clk);
    end
    
  end
  endtask
  
  // 写入多个组的数据
  task write_multiple_groups;
    input [15:0] fp16_a [0:31];
    input [15:0] fp16_b [0:31];
    input integer reg_bank;
    
    reg [127:0] test_s0, test_s1, test_s2, test_s3;
    reg [127:0] test_s4, test_s5, test_s6, test_s7;
    integer i;
  begin
    // 为所有组准备数据
    test_s0 = 128'h0; test_s1 = 128'h0; test_s2 = 128'h0; test_s3 = 128'h0;
    test_s4 = 128'h0; test_s5 = 128'h0; test_s6 = 128'h0; test_s7 = 128'h0;
    
    for (i = 0; i < NUM_GROUPS; i = i + 1) begin
      test_s0[i*4 +: 4] = fp16_a[i][3:0];   test_s1[i*4 +: 4] = fp16_a[i][7:4];
      test_s2[i*4 +: 4] = fp16_a[i][11:8];  test_s3[i*4 +: 4] = fp16_a[i][15:12];
      test_s4[i*4 +: 4] = fp16_b[i][3:0];   test_s5[i*4 +: 4] = fp16_b[i][7:4];
      test_s6[i*4 +: 4] = fp16_b[i][11:8];  test_s7[i*4 +: 4] = fp16_b[i][15:12];
    end
    
    // 写入s0-s3寄存器
    dvr_fp16mul_s0 = test_s0; dvr_fp16mul_s1 = test_s1;
    dvr_fp16mul_s2 = test_s2; dvr_fp16mul_s3 = test_s3;
    cru_fp16mul_s0123 = {1'b1, reg_bank[0]}; @(posedge clk);
    cru_fp16mul_s0123 = 2'b00; @(posedge clk);
    
    // 分别写入s4,s5,s6,s7寄存器
    dvr_fp16mul_s4567 = test_s4; cru_fp16mul_s4567 = {1'b1, reg_bank[0], 2'b00}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
    dvr_fp16mul_s4567 = test_s5; cru_fp16mul_s4567 = {1'b1, reg_bank[0], 2'b01}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
    dvr_fp16mul_s4567 = test_s6; cru_fp16mul_s4567 = {1'b1, reg_bank[0], 2'b10}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
    dvr_fp16mul_s4567 = test_s7; cru_fp16mul_s4567 = {1'b1, reg_bank[0], 2'b11}; @(posedge clk); cru_fp16mul_s4567 = 4'b0000; @(posedge clk);
    
    // 执行乘法运算
    cru_fp16mul = {1'b1, reg_bank[0], reg_bank[0]}; @(posedge clk); // 发起指令
    cru_fp16mul = 3'b000; @(posedge clk);                          
  end
  endtask
  
  // 从输出寄存器重构结果
  function [31:0] reconstruct_result;
    input integer group_id;
  begin
    reconstruct_result = {
      dr_fp16mul_d7[group_id*4 +: 4], dr_fp16mul_d6[group_id*4 +: 4],
      dr_fp16mul_d5[group_id*4 +: 4], dr_fp16mul_d4[group_id*4 +: 4],
      dr_fp16mul_d3[group_id*4 +: 4], dr_fp16mul_d2[group_id*4 +: 4],
      dr_fp16mul_d1[group_id*4 +: 4], dr_fp16mul_d0[group_id*4 +: 4]
    };
  end
  endfunction
  
  // 固定测试任务
  task test_with_softfloat;
    input [15:0] fp16_a;
    input [15:0] fp16_b;
    input integer group_id;
    input integer reg_bank;
    
    reg [31:0] actual_result;
  begin
    clear_softfloat_flags();
    expected_fp32 = fp16_inputs_mul_to_fp32_softfloat(fp16_a, fp16_b);
    softfloat_flags = get_softfloat_flags();
    write_fp16_to_registers(fp16_a, fp16_b, group_id, reg_bank, 1'b1);
    actual_result = reconstruct_result(group_id);
    if (verify_result(fp16_a, fp16_b, actual_result, expected_fp32, test_count, "固定"))
      pass_count = pass_count + 1;
    else
      fail_count = fail_count + 1;
  end
  endtask
  
  // 随机测试任务
  task random_test;
    input integer num_tests;
    
    reg [15:0] random_fp16_a;
    reg [15:0] random_fp16_b;
    reg [31:0] actual_result;
    integer i;
  begin
    $display("开始随机测试 (%0d次)...", num_tests);
    for (i = 0; i < num_tests; i = i + 1) begin
      random_fp16_a = $random & 16'hffff;
      random_fp16_b = $random & 16'hffff;
      clear_softfloat_flags();
      expected_fp32 = fp16_inputs_mul_to_fp32_softfloat(random_fp16_a, random_fp16_b);
      softfloat_flags = get_softfloat_flags();
      write_fp16_to_registers(random_fp16_a, random_fp16_b, 0, 0, 1'b1);
      actual_result = reconstruct_result(0);
      if (verify_result(random_fp16_a, random_fp16_b, actual_result, expected_fp32, i, "随机"))
        pass_count = pass_count + 1;
      else
        fail_count = fail_count + 1;
    end
    $display("随机测试完成");
  end
  endtask

  // 32组同时随机测试任务 
  task test_32groups_simultaneous_random;
    input integer num_tests;
    
    reg [15:0] fp16_a_array [0:31];
    reg [15:0] fp16_b_array [0:31];
    reg [31:0] expected_array [0:31];
    reg [31:0] actual_array [0:31];
    
    reg [127:0] s0_data, s1_data, s2_data, s3_data;
    reg [127:0] s4_data, s5_data, s6_data, s7_data;
    
    integer i, j;
    integer passed_in_test, failed_in_test;
  begin
    $display("开始32组同时随机测试 (%0d轮，每轮32组)...", num_tests);
    
    for (i = 0; i < num_tests; i = i + 1) begin
      passed_in_test = 0;
      failed_in_test = 0;
      
      for (j = 0; j < 32; j = j + 1) begin
        fp16_a_array[j] = $random & 16'hffff;
        fp16_b_array[j] = $random & 16'hffff;
        
        // 计算期望结果
        clear_softfloat_flags();
        expected_array[j] = fp16_inputs_mul_to_fp32_softfloat(fp16_a_array[j], fp16_b_array[j]);
      end
      
      // 将32组数据组装成128位寄存器值
      s0_data = 128'h0; s1_data = 128'h0; s2_data = 128'h0; s3_data = 128'h0;
      s4_data = 128'h0; s5_data = 128'h0; s6_data = 128'h0; s7_data = 128'h0;
      
      for (j = 0; j < 32; j = j + 1) begin
        s0_data[j*4 +: 4] = fp16_a_array[j][3:0];    
        s1_data[j*4 +: 4] = fp16_a_array[j][7:4];    
        s2_data[j*4 +: 4] = fp16_a_array[j][11:8];   
        s3_data[j*4 +: 4] = fp16_a_array[j][15:12];  
        s4_data[j*4 +: 4] = fp16_b_array[j][3:0];    
        s5_data[j*4 +: 4] = fp16_b_array[j][7:4];    
        s6_data[j*4 +: 4] = fp16_b_array[j][11:8];   
        s7_data[j*4 +: 4] = fp16_b_array[j][15:12];  
      end
      
      // 第1步：写入S0-S3寄存器（FP16_A数据）
      dvr_fp16mul_s0 = s0_data;
      dvr_fp16mul_s1 = s1_data;
      dvr_fp16mul_s2 = s2_data;
      dvr_fp16mul_s3 = s3_data;
      cru_fp16mul_s0123 = 2'b10;  // 有效写入到寄存器组0
      @(posedge clk);
      cru_fp16mul_s0123 = 2'b00;
      
      // 第2步：写入S4寄存器（FP16_B的低4位）
      dvr_fp16mul_s4567 = s4_data;
      cru_fp16mul_s4567 = 4'b1000;  // 有效写入S4到寄存器组0
      @(posedge clk);
      cru_fp16mul_s4567 = 4'b0000;
      
      // 第3步：写入S5寄存器（FP16_B的次低4位）
      dvr_fp16mul_s4567 = s5_data;
      cru_fp16mul_s4567 = 4'b1001;  // 有效写入S5到寄存器组0
      @(posedge clk);
      cru_fp16mul_s4567 = 4'b0000;
      
      // 第4步：写入S6寄存器（FP16_B的次高4位）
      dvr_fp16mul_s4567 = s6_data;
      cru_fp16mul_s4567 = 4'b1010;  // 有效写入S6到寄存器组0
      @(posedge clk);
      cru_fp16mul_s4567 = 4'b0000;
      
      // 第5步：写入S7寄存器（FP16_B的高4位）
      dvr_fp16mul_s4567 = s7_data;
      cru_fp16mul_s4567 = 4'b1011;  // 有效写入S7到寄存器组0
      @(posedge clk);
      cru_fp16mul_s4567 = 4'b0000;
      
      // 第6步：乘法运算指令
      cru_fp16mul = 3'b100;  // 有效乘法指令，使用寄存器组0*寄存器组0
      @(posedge clk);
      cru_fp16mul = 3'b000;
      
      // 第7步：读取结果
      @(posedge clk);  // 等待1个时钟周期让结果稳定
      
      // 从输出寄存器读取所有32组的结果
      for (j = 0; j < 32; j = j + 1) begin
        actual_array[j] = {
          dr_fp16mul_d7[j*4 +: 4], dr_fp16mul_d6[j*4 +: 4],
          dr_fp16mul_d5[j*4 +: 4], dr_fp16mul_d4[j*4 +: 4],
          dr_fp16mul_d3[j*4 +: 4], dr_fp16mul_d2[j*4 +: 4],
          dr_fp16mul_d1[j*4 +: 4], dr_fp16mul_d0[j*4 +: 4]
        };
        
        // 验证每组的结果
        if (verify_result(fp16_a_array[j], fp16_b_array[j], actual_array[j], expected_array[j], test_count, "32组同时")) begin
          passed_in_test = passed_in_test + 1;
          pass_count = pass_count + 1;
        end else begin
          failed_in_test = failed_in_test + 1;
          fail_count = fail_count + 1;
        end
      end
      
      $display("第%0d轮32组同时测试完成：通过%0d组，失败%0d组", i+1, passed_in_test, failed_in_test);
    end
    
    $display("32组同时随机测试完成");
  end
  endtask
  
  // 固定测试任务 
  task fixed_test;
  begin
    $display("开始固定测试...");
    // 基本规格化数测试
    test_with_softfloat (16'h3c00, 16'h3c00, 1, 0);// 测试 1: 1.0 * 1.0 = 1.0
    test_with_softfloat (16'h4000, 16'h3c00, 2, 0);// 测试 2: 2.0 * 1.0 = 2.0
    test_with_softfloat (16'h3c00, 16'h4000, 3, 0);// 测试 3: 1.0 * 2.0 = 2.0
    test_with_softfloat (16'h7bff, 16'h3c00, 4, 0);// 测试 4: max_norm * 1.0
    test_with_softfloat (16'h0400, 16'h3c00, 5, 0);// 测试 5: min_norm * 1.0

    // 无穷大测试
    test_with_softfloat (16'h7c00, 16'h3c00, 6, 0);// 测试 6: +inf * 1.0 = +inf
    test_with_softfloat (16'h3c00, 16'h7c00, 7, 0);// 测试 7: 1.0 * +inf = +inf
    test_with_softfloat (16'h7c00, 16'h7c00, 8, 0);// 测试 8: +inf * +inf = +inf

    // NaN 测试
    test_with_softfloat (16'h7c01, 16'h3c00, 1, 1);// 测试 9: NaN * 1.0 = NaN
    test_with_softfloat (16'h3c00, 16'h7c01, 2, 1);// 测试 10: 1.0 * NaN = NaN

    // 零测试
    test_with_softfloat (16'h0000, 16'h3c00, 3, 1);// 测试 11: +0 * 1.0 = +0
    test_with_softfloat (16'h3c00, 16'h0000, 4, 1);// 测试 12: 1.0 * +0 = +0
    test_with_softfloat (16'h8000, 16'h3c00, 5, 1);// 测试 13: -0 * 1.0 = -0

    // 非规格化数测试
    test_with_softfloat (16'h0001, 16'h3c00, 6, 1);// 测试 14: min_denorm * 1.0
    test_with_softfloat (16'h03ff, 16'h3c00, 7, 1);// 测试 15: max_denorm * 1.0

    // 更多规格化数测试
    test_with_softfloat (16'h4400, 16'h4400, 8, 1);// 测试 16: 4.0 * 4.0 = 16.0
    test_with_softfloat (16'h4400, 16'h4500, 0, 0);// 测试 17: 4.0 * 5.0 = 20.0
    test_with_softfloat (16'h4400, 16'h3e00, 0, 0);// 测试 18: 4.0 * 1.5 = 6.0
    test_with_softfloat (16'h3800, 16'h3800, 0, 0);// 测试 19: 0.5 * 0.5 = 0.25
    test_with_softfloat (16'h4400, 16'hc000, 0, 0);// 测试 20: 4.0 * -2.0 = -8.0

    // 负数测试
    test_with_softfloat (16'hbc00, 16'hbc00, 0, 0);// 测试 21: -1.0 * -1.0 = 1.0
    test_with_softfloat (16'h5400, 16'h5400, 0, 0);// 测试 22: 64.0 * 64.0 = 4096.0
    test_with_softfloat (16'h4800, 16'h3400, 0, 0);// 测试 23: 8.0 * 0.25 = 2.0

    // 边界条件测试
    test_with_softfloat (16'h0400, 16'h0400, 0, 0);// 测试 24: min_norm * min_norm -> denorm result
    test_with_softfloat (16'h7800, 16'h0400, 0, 0);// 测试 25: 2^15 * min_norm
    test_with_softfloat (16'h0400, 16'h7800, 0, 0);// 测试 26: min_norm * 2^15
    test_with_softfloat (16'h7800, 16'h7800, 0, 0);// 测试 27: 2^15 * 2^15 = +inf
    test_with_softfloat (16'h7800, 16'h0800, 0, 0);// 测试 28: 2^15 * 2^-13 = 4.0

    // 尾数边界情况
    test_with_softfloat (16'h3bff, 16'h4000, 0, 0);// 测试 29: (1.0-eps) * 2.0
    test_with_softfloat (16'h3c01, 16'h3c01, 0, 0);// 测试 30: (1.0+eps) * (1.0+eps)
    
    // 边界和特殊值组合测试
    test_with_softfloat (16'h0001, 16'h0001, 0, 0); // min_denorm * min_denorm -> underflow to zero
    test_with_softfloat (16'h0400, 16'h8400, 0, 0); // min_norm * -min_norm -> denormalized result
    test_with_softfloat (16'h7c00, 16'h0000, 0, 0); // +inf * +0 -> NaN
    test_with_softfloat (16'hfc00, 16'h7c00, 0, 0); // -inf * +inf -> -inf
    test_with_softfloat (16'h7c01, 16'h7c00, 0, 0); // NaN * inf -> NaN
    test_with_softfloat (16'h0000, 16'h8000, 0, 0); // +0 * -0 -> -0
    test_with_softfloat (16'h03ff, 16'h0400, 0, 0); // max_denorm * min_norm -> normalized result
    test_with_softfloat (16'h7bff, 16'h7bff, 0, 0); // max_norm * max_norm -> overflow to inf

    $display("固定测试完成");
  end
  endtask

  initial begin 

    set_softfloat_rounding_mode(SOFTFLOAT_ROUND_NEAR_EVEN);
    
    // 初始化信号
    test_count = 0; pass_count = 0; fail_count = 0;
    dvr_fp16mul_s0 = 128'h0; dvr_fp16mul_s1 = 128'h0;
    dvr_fp16mul_s2 = 128'h0; dvr_fp16mul_s3 = 128'h0;
    dvr_fp16mul_s4567 = 128'h0;
    cru_fp16mul_s0123 = 2'b00;
    cru_fp16mul_s4567 = 4'b0000;
    cru_fp16mul = 3'b000;
    
    // 复位系统
    reset_system();
    // 执行固定测试
    fixed_test();
    // 执行无效指令测试
    test_idle_instruction(16'h3c00, 16'h3c00, 0, 0);
    // 执行所有寄存器组组合测试
    test_all_bank_combinations(16'h3c00, 16'h4000, 16'hbc00, 16'h4400, 0); 
    // 执行32组同时随机测试
    test_32groups_simultaneous_random(5); 
    // 执行随机测试
    random_test(random_num);
    
    // 等待所有操作完成
    repeat (5) @(posedge clk);
    
    // 显示测试总结
    $display("=== 测试完成 ===");
    $display("总测试数: %0d", test_count);
    $display("通过: %0d", pass_count);
    $display("失败: %0d", fail_count);
    if (test_count > 0)
        $display("通过率: %0.1f%%", (pass_count * 100.0) / test_count);
    
    if (fail_count == 0) begin
      $display("*** 所有测试通过! ***");
    end else begin
      $display("*** 有测试失败! ***");
    end
    
    $finish;
  end

endmodule
