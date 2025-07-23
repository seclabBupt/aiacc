`timescale 1ns/1ps
module tb_logical_unit();

reg clk;  // 修正：原为dk，应改为clk
reg rst_n;
reg logical_vld_i;
reg [3:0] logical_op_i;
reg logical_precision_i;  // 0:16-bit, 1:32-bit
reg [31:0] logical_src0_i;
reg [31:0] logical_src1_i;
reg [2:0] fpadd_status_i;
// 新增移位方向信号
reg shift_dir_i;  // 0=左移, 1=右移

wire logical_done_o;
wire [31:0] logical_dst_o;

// DPI导入声明（保持不变）
import "DPI-C" function void set_softfloat_rounding_mode(input byte unsigned mode);
import "DPI-C" function void clear_softfloat_flags();
import "DPI-C" function byte unsigned get_softfloat_flags();
import "DPI-C" function byte unsigned fp32_compare_softfloat(input bit[31:0] a, input bit[31:0] b);
import "DPI-C" function byte unsigned fp16_compare_softfloat(input bit[15:0] a, input bit[15:0] b);

// 时钟生成
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// 复位控制
initial begin
    rst_n = 0;
    #20 rst_n = 1;
end

// 实例化待测模块（添加shift_dir_i端口）
logical_unit dut(
    .clk(clk),
    .rst_n(rst_n),
    .logical_vld_i(logical_vld_i),
    .logical_op_i(logical_op_i),
    .logical_precision_i(logical_precision_i),
    .logical_src0_i(logical_src0_i),  // 修正：原为logical_srcO_i
    .logical_src1_i(logical_src1_i),
    .fpadd_status_i(fpadd_status_i),
    .shift_dir_i(shift_dir_i),        // 新增方向信号
    .logical_done_o(logical_done_o),
    .logical_dst_o(logical_dst_o)
);

// 操作类型定义（保持原样）
localparam [3:0] 
    OP_AND         = 4'b0000,
    OP_OR          = 4'b0001,
    OP_XOR         = 4'b0010,
    OP_NOT         = 4'b0011,
    OP_COPY        = 4'b0100,
    OP_SELECT_GT   = 4'b0101,
    OP_SELECT_EQ   = 4'b0110,
    OP_SELECT_LS   = 4'b0111,
    OP_LOGIC_SHIFT = 4'b1000,
    OP_ARITH_SHIFT = 4'b1001,
    OP_ROT_SHIFT   = 4'b1010;

// 测试变量
integer pass_count = 0;
integer fail_count = 0;
integer total_tests = 0;
integer softfloat_flag_count = 0;

// 浮点数值常量定义（保持原样）
localparam [31:0]
    FP32_1_0     = 32'h3F800000,  // 1.0
    FP32_2_0     = 32'h40000000,  // 2.0
    FP32_3_0     = 32'h40400000,  // 3.0
    FP32_INF     = 32'h7F800000,  // +Inf
    FP32_NEG_INF = 32'hFF800000,  // -Inf
    FP32_NAN     = 32'h7FC00000;  // NaN

localparam [15:0]
    FP16_1_0 = 16'h3C00,
    FP16_2_0 = 16'h4000,
    FP16_3_0 = 16'h4200;

// 测试任务（添加shift_dir参数）
task test_case;
    input [3:0] op;
    input precision;
    input [31:0] src0;
    input [31:0] src1;
    input string test_name;
    input shift_dir;  // 新增：移位方向控制
    
    // 内部变量声明
    reg [31:0] softfloat_expected;
    reg [2:0] softfloat_status;
    byte unsigned cmp_result;
    reg [31:0] shift_val;
    reg [31:0] rotated;
    reg [31:0] double_data;
    reg [31:0] temp_shifted;
    reg [15:0] rotated_16;
    reg signed [15:0] signed_temp;
    
begin
    total_tests = total_tests + 1;
    clear_softfloat_flags();
    
    // 设置移位方向信号
    shift_dir_i = shift_dir;  // 新增
    
    case(op)
        // 选择操作（保持原样）
        OP_SELECT_GT, OP_SELECT_EQ, OP_SELECT_LS: begin
            if(precision) begin
                cmp_result = fp32_compare_softfloat(src0, src1);
            end else begin
                cmp_result = fp16_compare_softfloat(src0[15:0], src1[15:0]);
            end
            softfloat_status = {
                (cmp_result == 2),  // GT
                (cmp_result == 0),  // EQ
                (cmp_result == 1)   // LS
            };
            
            case(op)
                OP_SELECT_GT: softfloat_expected = softfloat_status[2] ? src0 : src1;
                OP_SELECT_EQ: softfloat_expected = softfloat_status[1] ? src0 : src1;
                OP_SELECT_LS: softfloat_expected = softfloat_status[0] ? src0 : src1;
            endcase
        end
        
        // 逻辑移位操作（添加方向支持）
        OP_LOGIC_SHIFT: begin
            softfloat_status = 3'b000;
            shift_val = src1;
            
            if(precision) begin  // 32-bit
                shift_val = shift_val & 32'h1F;  // 取模32
                if(shift_dir)  // 右移
                    softfloat_expected = src0 >> shift_val;
                else           // 左移
                    softfloat_expected = src0 << shift_val;
            end else begin      // 16-bit
                shift_val = shift_val & 32'hF;  // 取模16
                if(shift_dir)  // 右移
                    softfloat_expected = {16'b0, src0[15:0] >> shift_val[3:0]};
                else           // 左移
                    softfloat_expected = {16'b0, src0[15:0] << shift_val[3:0]};
            end
        end
        
        // 算术移位操作（添加方向支持和符号扩展）
        OP_ARITH_SHIFT: begin
            softfloat_status = 3'b000;
            shift_val = src1;
            
            if(precision) begin  // 32-bit
                shift_val = shift_val & 32'h1F;
                if(shift_dir)  // 算术右移
                    softfloat_expected = $signed(src0) >>> shift_val;
                else           // 算术左移
                    softfloat_expected = $signed(src0) <<< shift_val;
            end else begin      // 16-bit
                shift_val = shift_val & 32'hF;
                signed_temp = $signed(src0[15:0]);
                
                if(shift_dir) begin  // 算术右移
                    signed_temp = signed_temp >>> shift_val[3:0];
                    // 符号扩展
                    softfloat_expected = signed_temp[15] ? {16'hFFFF, signed_temp} : {16'h0, signed_temp};
                end else begin       // 算术左移
                    signed_temp = signed_temp <<< shift_val[3:0];
                    softfloat_expected = {16'b0, signed_temp};
                end
            end
        end
        
        // 旋转移位操作（保持原样）
        OP_ROT_SHIFT: begin
            softfloat_status = 3'b000;
            shift_val = src1;
            
            if(precision) begin
                shift_val = shift_val % 32;
                rotated = (src0 >> shift_val) | (src0 << (32 - shift_val));
                softfloat_expected = rotated;
            end else begin
                shift_val = shift_val % 16;
                double_data = {src0[15:0], src0[15:0]};
                temp_shifted = double_data >> shift_val;
                rotated_16 = temp_shifted[15:0];
                softfloat_expected = {16'b0, rotated_16};
            end
        end
        
        // 基本逻辑操作（保持原样）
        default: begin
            softfloat_status = 3'b000;
            case(op)
                OP_AND:  softfloat_expected = precision ? (src0 & src1) : {16'b0, src0[15:0] & src1[15:0]};
                OP_OR:   softfloat_expected = precision ? (src0 | src1) : {16'b0, src0[15:0] | src1[15:0]};
                OP_XOR:  softfloat_expected = precision ? (src0 ^ src1) : {16'b0, src0[15:0] ^ src1[15:0]};
                OP_NOT:  softfloat_expected = precision ? (~src0) : {16'b0, ~src0[15:0]};
                OP_COPY: softfloat_expected = precision ? src0 : {16'b0, src0[15:0]};
                default: softfloat_expected = 32'b0;
            endcase
        end
    endcase
    
    // 记录SoftFloat异常标志
    if(get_softfloat_flags() != 0) begin
        softfloat_flag_count = softfloat_flag_count + 1;
        $display("SoftFloat Flags in '%s': 0x%04X", test_name, get_softfloat_flags());
    end
    
    @(posedge clk);
    logical_vld_i = 1;
    logical_op_i = op;
    logical_precision_i = precision;
    logical_src0_i = src0;
    logical_src1_i = src1;
    fpadd_status_i = softfloat_status;
    
    @(posedge clk);
    logical_vld_i = 0;
    wait(logical_done_o);
    #1;  // 等待信号稳定
    
    // 显示结果
    $display("[TEST] '%s' at %0tns", test_name, $time);
    $display(" Precision: %s", precision ? "32-bit" : "16-bit");
    $display(" Input: S0=%08X, S1=%08X", src0, src1);
    
    if(op >= OP_SELECT_GT && op <= OP_SELECT_LS) begin
        $display(" SoftFloat Status: GT=%b, EQ=%b, LS=%b",
                softfloat_status[2], softfloat_status[1], softfloat_status[0]);
    end
    
    $display(" Expected: %08X, Got: %08X", softfloat_expected, logical_dst_o);
    
    if(logical_dst_o !== softfloat_expected) begin
        $display("[FAIL] Mismatch detected!");
        fail_count = fail_count + 1;
    end else begin
        $display("[PASS] Result matches");
        pass_count = pass_count + 1;
    end
end
endtask

// 主测试流程（添加方向测试用例）
initial begin
    // 初始化SoftFloat
    set_softfloat_rounding_mode(0);  // 最近偶数舍入
    
    // 等待复位完成
    @(posedge rst_n);
    #10;
    
    $display("\n===== Selection Operation Tests (32-bit) =====");
    test_case(OP_SELECT_GT, 1, 32'h40400000, 32'h3F800000, "SELECT_GT:3.0>1.0", 0);
    test_case(OP_SELECT_EQ, 1, 32'h40000000, 32'h40000000, "SELECT_EQ:2.0==2.0", 0);
    test_case(OP_SELECT_LS, 1, 32'h3F800000, 32'h40400000, "SELECT_LS:1.0<3.0", 0);
    
    $display("\n===== Selection Operation Tests (16-bit) =====");
    test_case(OP_SELECT_GT, 0, 32'h00004200, 32'h00003C00, "SELECT_GT:3.0>1.0", 0);
    test_case(OP_SELECT_EQ, 0, 32'h00004000, 32'h00004000, "SELECT_EQ:2.0==2.0", 0);
    
    $display("\n===== Boundary Tests (Inf/NaN) =====");
    test_case(OP_SELECT_GT, 1, 32'h7F800000, 32'h7F800000, "SELECT_GT: Inf==Inf", 0);
    test_case(OP_SELECT_EQ, 1, 32'h7F800000, 32'hFF800000, "SELECT_EQ:+Inf!=-Inf", 0);
    test_case(OP_SELECT_LS, 1, 32'h7FC00000, 32'h7F800000, "SELECT_LS: NaN comparison", 0);
    
    $display("\n===== Basic Logic Tests (32-bit) =====");
    test_case(OP_AND, 1, 32'hA5A5A5A5, 32'h0F0F0F0F, "32-bit AND", 0);
    test_case(OP_OR, 1, 32'hA5A5A5A5, 32'h0F0F0F0F, "32-bit OR", 0);
    test_case(OP_XOR, 1, 32'hA5A5A5A5, 32'h0F0F0F0F, "32-bit XOR", 0);
    test_case(OP_NOT, 1, 32'hA5A5A5A5, 32'h00000000, "32-bit NOT", 0);
    test_case(OP_COPY, 1, 32'hDEADBEEF, 32'h00000000, "32-bit COPY", 0);
    
    $display("\n===== Basic Logic Tests (16-bit) =====");
    test_case(OP_AND, 0, 32'hA5A5FFFF, 32'h0F0F1234, "16-bit AND", 0);
    test_case(OP_OR, 0, 32'hA5A5FFFF, 32'h0F0F1234, "16-bit OR", 0);
    test_case(OP_XOR, 0, 32'hA5A5FFFF, 32'h0F0F1234, "16-bit XOR", 0);
    test_case(OP_NOT, 0, 32'hA5A5FFFF, 32'h00000000, "16-bit NOT", 0);
    test_case(OP_COPY, 0, 32'hDEADBEEF, 32'h00000000, "16-bit COPY", 0);
    
    // ========= 新增方向测试用例 =========
    $display("\n===== Shift Operation Tests (32-bit) =====");
    // 逻辑移位
    test_case(OP_LOGIC_SHIFT, 1, 32'hF0F0F0F0, 32'h00000004, "Logic Shift Left 4", 0);
    test_case(OP_LOGIC_SHIFT, 1, 32'hF0F0F0F0, 32'h00000004, "Logic Shift Right 4", 1);
    test_case(OP_LOGIC_SHIFT, 1, 32'h0F0F0F0F, 32'h00000008, "Logic Shift Left 8", 0);
    test_case(OP_LOGIC_SHIFT, 1, 32'h0F0F0F0F, 32'h00000008, "Logic Shift Right 8", 1);
    
    // 算术移位
    test_case(OP_ARITH_SHIFT, 1, 32'hF0F0F0F0, 32'h00000004, "Arith Shift Right 4 (negative)", 1);
    test_case(OP_ARITH_SHIFT, 1, 32'hF0F0F0F0, 32'h00000004, "Arith Shift Left 4 (negative)", 0);
    test_case(OP_ARITH_SHIFT, 1, 32'h70F0F0F0, 32'h00000004, "Arith Shift Right 4 (positive)", 1);
    test_case(OP_ARITH_SHIFT, 1, 32'h70F0F0F0, 32'h00000004, "Arith Shift Left 4 (positive)", 0);
    
    // 旋转移位（不需要方向）
    test_case(OP_ROT_SHIFT, 1, 32'h12345678, 32'h00000008, "Rotate Shift 8", 0);
    test_case(OP_ROT_SHIFT, 1, 32'h12345678, 32'h00000010, "Rotate Shift 16", 0);
    
    $display("\n===== Shift Operation Tests (16-bit) =====");
    // 逻辑移位
    test_case(OP_LOGIC_SHIFT, 0, 32'h0000F0F0, 32'h00000004, "16b Logic Shift Left 4", 0);
    test_case(OP_LOGIC_SHIFT, 0, 32'h0000F0F0, 32'h00000004, "16b Logic Shift Right 4", 1);
    
    // 算术移位（带符号扩展）
    test_case(OP_ARITH_SHIFT, 0, 32'h0000F0F0, 32'h00000004, "16b Arith Shift Right 4 (negative)", 1);
    test_case(OP_ARITH_SHIFT, 0, 32'h0000F0F0, 32'h00000004, "16b Arith Shift Left 4 (negative)", 0);
    test_case(OP_ARITH_SHIFT, 0, 32'h000070F0, 32'h00000004, "16b Arith Shift Right 4 (positive)", 1);
    test_case(OP_ARITH_SHIFT, 0, 32'h000070F0, 32'h00000004, "16b Arith Shift Left 4 (positive)", 0);
    
    // 旋转移位
    test_case(OP_ROT_SHIFT, 0, 32'h00001234, 32'h00000004, "16b Rotate Shift 4", 0);
    test_case(OP_ROT_SHIFT, 0, 32'h00001234, 32'h0000000C, "16b Rotate Shift 12", 0);
    
    $display("\n===== Boundary Tests =====");
    // 方向边界测试
    test_case(OP_LOGIC_SHIFT, 1, 32'h80000000, 32'h0000001F, "32b Logic Shift Left 31", 0);
    test_case(OP_LOGIC_SHIFT, 1, 32'h80000000, 32'h0000001F, "32b Logic Shift Right 31", 1);
    test_case(OP_ARITH_SHIFT, 1, 32'h80000000, 32'h0000001F, "32b Arith Shift Right 31", 1);
    test_case(OP_ARITH_SHIFT, 1, 32'h80000000, 32'h0000001F, "32b Arith Shift Left 31", 0);
    
    // 16位符号扩展边界测试
    test_case(OP_ARITH_SHIFT, 0, 32'h00008001, 32'h00000001, "16b Arith Shift Right 1 (sign extend)", 1);
    test_case(OP_ARITH_SHIFT, 0, 32'h00000001, 32'h00000001, "16b Arith Shift Left 1 (no sign)", 0);
    
    // 完成测试
    #20;
    $display("\n===== Test Summary =====");
    $display(" Total tests: %0d", total_tests);
    $display(" Passed: %0d, Failed: %0d", pass_count, fail_count);
    $display(" SoftFloat Flags detected: %0d", softfloat_flag_count);
    
    if(fail_count == 0)
        $display("SUCCESS: All tests passed!");
    else
        $display("FAILURE: Some tests failed!");
    
    $finish;
end

// 波形记录
initial begin
    $dumpfile("tb_logical_unit.vcd");
    $dumpvars(0, tb_logical_unit);
end

endmodule