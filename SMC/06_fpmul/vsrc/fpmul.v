// 支持 fp16 和 fp32 浮点数乘法运算
module fpmul (
    // 微指令控制信号
    input  wire        inst_valid,           // 指令有效信号：1'b1有效，1'b0无效
    input  wire        src_precision,        // 源寄存器精度：1'b0为16bit，1'b1为32bit
    input  wire        dst_precision,        // 目的寄存器精度：1'b0为16bit，1'b1为32bit
    
    // 数据输入端口
    input  wire [31:0] dvr_fpmul_s0,         // FPMUL第一输入寄存器
    input  wire [31:0] dvr_fpmul_s1,         // FPMUL第二输入寄存器
    
    // 数据输出端口
    output wire [31:0] dr_fpmul_d            // FPMUL输出寄存器
);

    // 内部信号定义
    wire [15:0] fp16_a, fp16_b;              // fp16 操作数
    wire [31:0] fp32_a, fp32_b;              // fp32 操作数
    wire [15:0] fp16_result;                 // fp16 乘法结果
    wire [31:0] fp32_result;                 // fp32 乘法结果
    
    // 从输入寄存器提取操作数
    assign fp16_a = dvr_fpmul_s0[15:0];      // 提取fp16第一操作数
    assign fp16_b = dvr_fpmul_s1[15:0];      // 提取fp16第二操作数
    assign fp32_a = dvr_fpmul_s0;            // fp32第一操作数
    assign fp32_b = dvr_fpmul_s1;            // fp32第二操作数

    // FP16 乘法器实例化
    fp16_multiplier u_fp16_mul (
        .a(fp16_a),
        .b(fp16_b),
        .result(fp16_result)
    );
    
    // FP32 乘法器实例化
    fp32_multiplier u_fp32_mul (
        .a(fp32_a),
        .b(fp32_b),
        .result(fp32_result)
    );
    
    // 输出多路选择器
    assign dr_fpmul_d = inst_valid ? (src_precision ? fp32_result : {16'h0000, fp16_result}) : 32'h00000000;

endmodule

// FP16 浮点乘法器子模块
module fp16_multiplier (
    input  wire [15:0] a,                    // fp16 第一操作数
    input  wire [15:0] b,                    // fp16 第二操作数
    output wire [15:0] result                // fp16 乘法结果
);

    // FP16 格式定义
    localparam FP16_EXP_WIDTH = 5;
    localparam FP16_MANT_WIDTH = 10;
    localparam FP16_BIAS = 15;

    // 提取FP16字段
    wire sign_a = a[15];
    wire sign_b = b[15];
    wire [FP16_EXP_WIDTH-1:0] exp_a = a[14:10];
    wire [FP16_EXP_WIDTH-1:0] exp_b = b[14:10];
    wire [FP16_MANT_WIDTH-1:0] mant_a = a[9:0];
    wire [FP16_MANT_WIDTH-1:0] mant_b = b[9:0];

    // 分析输入值类型
    wire a_is_zero = (exp_a == 0) && (mant_a == 0);
    wire b_is_zero = (exp_b == 0) && (mant_b == 0);
    wire output_is_zero = a_is_zero || b_is_zero;

    wire a_is_inf = (exp_a == 5'b11111) && (mant_a == 0);
    wire b_is_inf = (exp_b == 5'b11111) && (mant_b == 0);
    wire output_is_inf = (a_is_inf || b_is_inf) && !(a_is_zero || b_is_zero);

    wire a_is_nan = (exp_a == 5'b11111) && (mant_a != 0);
    wire b_is_nan = (exp_b == 5'b11111) && (mant_b != 0);
    
    wire inf_times_zero = (a_is_inf && b_is_zero) || (b_is_inf && a_is_zero);
    wire output_is_nan = a_is_nan || b_is_nan || inf_times_zero;

    wire a_is_denorm = (exp_a == 0) && (mant_a != 0);
    wire b_is_denorm = (exp_b == 0) && (mant_b != 0);

    // 输出符号位 = 输入符号位异或
    wire sign_out = sign_a ^ sign_b;

    // =========================== 处理尾数 =========================
    // 规格化数的隐含前导1，非规格化数则为0
    wire [FP16_MANT_WIDTH:0] mant_a_with_hidden = a_is_denorm ? {1'b0, mant_a} : {1'b1, mant_a};
    wire [FP16_MANT_WIDTH:0] mant_b_with_hidden = b_is_denorm ? {1'b0, mant_b} : {1'b1, mant_b};

    // 执行尾数乘法，结果为22位
    wire [2*(FP16_MANT_WIDTH+1)-1:0] mant_product = mant_a_with_hidden * mant_b_with_hidden;

    // 改进的规格化检测
    wire normalize_shift = mant_product[2*(FP16_MANT_WIDTH+1)-1];

    // 表明是否需要处理非规格化数
    wire need_denorm_handling = a_is_denorm || b_is_denorm;
    
    // 前导零计算 - 使用LZC模块
    wire [4:0] leading_zeros;
    leading_zero_counter #(
        .DATA_WIDTH(22),
        .COUNT_WIDTH(5)
    ) lzc_fp16 (
        .data_in(mant_product),
        .leading_zeros(leading_zeros)
    );

    wire [21:0] shifted_mant = (mant_product == 0) ? 22'b0 :
                               (need_denorm_handling) ? ((leading_zeros >= 22) ? 22'b0 : (mant_product << leading_zeros)) :
                               (normalize_shift) ? (mant_product >> 1) :
                               mant_product;

    // final_mant的计算，加入舍入处理
    wire [FP16_MANT_WIDTH-1:0] base_mant = 
        (mant_product == 0) ? 10'b0 :
        need_denorm_handling ? 
            shifted_mant[20:11] :     // 取高10位
        normalize_shift ? 
            mant_product[20:11] :     // 取高10位
            mant_product[19:10];      // 取高10位
    
    // 舍入位和粘贴位计算 
    wire round_bit, sticky_bit;
    
    // 根据不同情况计算round和sticky位
    assign round_bit = need_denorm_handling ? shifted_mant[10] :
                       normalize_shift ? mant_product[10] : mant_product[9];
                       
    assign sticky_bit = need_denorm_handling ? (|shifted_mant[9:0]) :
                        normalize_shift ? (|mant_product[9:0]) : (|mant_product[8:0]);
    
    // IEEE 754标准舍入逻辑 (Round to Nearest Even)
    wire do_round = round_bit && (sticky_bit || base_mant[0]);
    wire [10:0] rounded_mant = base_mant + do_round;
    wire round_overflow = rounded_mant[10];  // 舍入溢出
    wire [9:0] final_mant = round_overflow ? rounded_mant[10:1] : rounded_mant[9:0];

    // ================================ 指数计算 ===========================
    // 非规格化数的无偏指数为-14（等于1-偏置）
    wire signed [7:0] exp_a_unbiased = a_is_denorm ? -14 : {3'b000, exp_a} - FP16_BIAS;
    wire signed [7:0] exp_b_unbiased = b_is_denorm ? -14 : {3'b000, exp_b} - FP16_BIAS;

    // 计算输出无偏指数
    wire signed [9:0] exp_out_unbiased = {{2{exp_a_unbiased[7]}}, exp_a_unbiased} + 
                                         {{2{exp_b_unbiased[7]}}, exp_b_unbiased};
    
    // 修改指数调整逻辑
    wire signed [9:0] exp_adjustment = 
        need_denorm_handling ? 
            (mant_product == 0) ? 10'd0 :
            -{{5{1'b0}}, leading_zeros} + {9'b0, shifted_mant[21]} :
        normalize_shift ? 10'd1 : 10'd0;

    // 添加FP16偏置得到偏置指数，考虑舍入溢出
    wire signed [9:0] exp_out_biased = exp_out_unbiased + exp_adjustment + FP16_BIAS + (round_overflow ? 10'd1 : 10'd0);

    // 处理指数溢出和下溢
    wire exp_overflow = exp_out_biased > 30;
    wire exp_underflow = exp_out_biased < 1;
    
    // 非规格化数结果的处理
    wire result_is_denormal = (exp_out_biased <= 0) && (exp_out_biased > -24) && (mant_product != 0);
    
    // 对于非规格化数结果，需要右移尾数
    wire [5:0] denorm_shift = result_is_denormal ? (1 - exp_out_biased) : 0;
    wire [22:0] temp_mant = {1'b1, final_mant, 12'b0}; // 扩展为23位进行移位
    wire [22:0] shifted_denorm = (denorm_shift >= 23) ? 23'b0 : (temp_mant >> denorm_shift);
    wire [9:0] denorm_final_mant = shifted_denorm[21:12]; // 取高10位
    
    // 最终尾数选择
    wire [FP16_MANT_WIDTH-1:0] computed_mant = result_is_denormal ? denorm_final_mant : final_mant;

    // 确保所有位都有明确的赋值
    wire [FP16_MANT_WIDTH-1:0] mant_out = 
        output_is_nan    ? {1'b1, {(FP16_MANT_WIDTH-1){1'b0}}} : // NaN (包括inf*0)
        output_is_zero   ? {FP16_MANT_WIDTH{1'b0}} :     // 零
        output_is_inf    ? {FP16_MANT_WIDTH{1'b0}} :     // 无穷大
        exp_overflow     ? {FP16_MANT_WIDTH{1'b0}} :     // 溢出到无穷大
        (exp_underflow && !result_is_denormal) ? {FP16_MANT_WIDTH{1'b0}} : // 下溢到零
        computed_mant;                                   // 正常情况或非规格化数

    // 最终指数值，考虑特殊情况
    wire [FP16_EXP_WIDTH-1:0] exp_out = 
        output_is_nan    ? 5'd31 :                 // NaN (包括inf*0)
        output_is_zero   ? 5'd0 :                  // 零
        output_is_inf    ? 5'd31 :                 // 无穷大
        exp_overflow     ? 5'd31 :                 // 溢出到无穷大
        (exp_underflow && !result_is_denormal) ? 5'd0 : // 下溢到零
        result_is_denormal ? 5'd0 :               // 非规格化数指数为0
        exp_out_biased[4:0];                       // 正常情况
    
    // 输出结果组装
    assign result = {sign_out, exp_out, mant_out};

endmodule

// FP32 浮点乘法器子模块
module fp32_multiplier (
    input  wire [31:0] a,                    // fp32 第一操作数
    input  wire [31:0] b,                    // fp32 第二操作数
    output wire [31:0] result                // fp32 乘法结果
);

    // FP32 格式定义
    localparam FP32_EXP_WIDTH = 8;
    localparam FP32_MANT_WIDTH = 23;
    localparam FP32_BIAS = 127;

    // 提取FP32字段
    wire sign_a = a[31];
    wire sign_b = b[31];
    wire [FP32_EXP_WIDTH-1:0] exp_a = a[30:23];
    wire [FP32_EXP_WIDTH-1:0] exp_b = b[30:23];
    wire [FP32_MANT_WIDTH-1:0] mant_a = a[22:0];
    wire [FP32_MANT_WIDTH-1:0] mant_b = b[22:0];

    // 分析输入值类型
    wire a_is_zero = (exp_a == 0) && (mant_a == 0);
    wire b_is_zero = (exp_b == 0) && (mant_b == 0);
    wire output_is_zero = a_is_zero || b_is_zero;

    wire a_is_inf = (exp_a == 8'b11111111) && (mant_a == 0);
    wire b_is_inf = (exp_b == 8'b11111111) && (mant_b == 0);
    wire output_is_inf = (a_is_inf || b_is_inf) && !(a_is_zero || b_is_zero);

    wire a_is_nan = (exp_a == 8'b11111111) && (mant_a != 0);
    wire b_is_nan = (exp_b == 8'b11111111) && (mant_b != 0);
    // 无穷大乘以0也产生NaN
    wire inf_times_zero = (a_is_inf && b_is_zero) || (b_is_inf && a_is_zero);
    wire output_is_nan = a_is_nan || b_is_nan || inf_times_zero;

    wire a_is_denorm = (exp_a == 0) && (mant_a != 0);
    wire b_is_denorm = (exp_b == 0) && (mant_b != 0);

    // 输出符号位 = 输入符号位异或
    wire sign_out = sign_a ^ sign_b;

    // =========================== 处理尾数 =========================
    // 规格化数的隐含前导1，非规格化数则为0
    wire [FP32_MANT_WIDTH:0] mant_a_with_hidden = a_is_denorm ? {1'b0, mant_a} : {1'b1, mant_a};
    wire [FP32_MANT_WIDTH:0] mant_b_with_hidden = b_is_denorm ? {1'b0, mant_b} : {1'b1, mant_b};

    // 执行尾数乘法，结果为48位
    wire [2*(FP32_MANT_WIDTH+1)-1:0] mant_product = mant_a_with_hidden * mant_b_with_hidden;

    // 改进的规格化检测
    wire normalize_shift = mant_product[2*(FP32_MANT_WIDTH+1)-1];

    // 表明是否需要处理非规格化数
    wire need_denorm_handling = a_is_denorm || b_is_denorm;
    
    // 前导零计算 - 使用LZC模块
    wire [5:0] leading_zeros;
    leading_zero_counter #(
        .DATA_WIDTH(48),
        .COUNT_WIDTH(6)
    ) lzc_fp32 (
        .data_in(mant_product),
        .leading_zeros(leading_zeros)
    );

    wire [47:0] shifted_mant = (mant_product == 0) ? 48'b0 :
                               (need_denorm_handling) ? ((leading_zeros >= 48) ? 48'b0 : (mant_product << leading_zeros)) :
                               (normalize_shift) ? (mant_product >> 1) :
                               mant_product;

    // final_mant的计算，加入舍入处理
    wire [FP32_MANT_WIDTH-1:0] base_mant = 
        (mant_product == 0) ? 23'b0 :
        need_denorm_handling ? 
            shifted_mant[46:24] :     // 取高23位
        normalize_shift ? 
            mant_product[46:24] :     // 取高23位
            mant_product[45:23];      // 取高23位
    
    // 舍入位和粘贴位计算 - 简化版本
    wire round_bit, sticky_bit;
    
    // 根据不同情况计算round和sticky位             刘老师无敌
    assign round_bit = need_denorm_handling ? shifted_mant[23] :
                       normalize_shift ? mant_product[23] : mant_product[22];
                       
    assign sticky_bit = need_denorm_handling ? (|shifted_mant[22:0]) :
                        normalize_shift ? (|mant_product[22:0]) : (|mant_product[21:0]);
    
    // IEEE 754标准舍入逻辑 (Round to Nearest Even)
    wire do_round = round_bit && (sticky_bit || base_mant[0]);
    wire [23:0] rounded_mant = base_mant + do_round;
    wire round_overflow = rounded_mant[23];  // 舍入溢出
    wire [22:0] final_mant = round_overflow ? rounded_mant[23:1] : rounded_mant[22:0];

    // ================================ 指数计算 ===========================
    // 非规格化数的无偏指数为-126（等于1-偏置）
    wire signed [9:0] exp_a_unbiased = a_is_denorm ? -126 : {2'b00, exp_a} - FP32_BIAS;
    wire signed [9:0] exp_b_unbiased = b_is_denorm ? -126 : {2'b00, exp_b} - FP32_BIAS;

    // 计算输出无偏指数
    wire signed [11:0] exp_out_unbiased = {{2{exp_a_unbiased[9]}}, exp_a_unbiased} + 
                                          {{2{exp_b_unbiased[9]}}, exp_b_unbiased};
    
    // 修改指数调整逻辑
    wire signed [11:0] exp_adjustment = 
        need_denorm_handling ? 
            (mant_product == 0) ? 12'd0 :
            -{{6{1'b0}}, leading_zeros} + {11'b0, shifted_mant[47]} :
        normalize_shift ? 12'd1 : 12'd0;

    // 添加FP32偏置得到偏置指数，考虑舍入溢出
    wire signed [11:0] exp_out_biased = exp_out_unbiased + exp_adjustment + FP32_BIAS + (round_overflow ? 12'd1 : 12'd0);

    // 处理指数溢出和下溢
    wire exp_overflow = exp_out_biased > 254;
    wire exp_underflow = exp_out_biased < 1;
    
    // 非规格化数结果的处理
    wire result_is_denormal = (exp_out_biased <= 0) && (exp_out_biased > -150) && (mant_product != 0);
    
    // 对于非规格化数结果，需要右移尾数
    wire [7:0] denorm_shift = result_is_denormal ? (1 - exp_out_biased) : 0;
    wire [46:0] temp_mant = {1'b1, final_mant, 23'b0}; // 扩展为47位进行移位
    wire [46:0] shifted_denorm = (denorm_shift >= 47) ? 47'b0 : (temp_mant >> denorm_shift);
    wire [22:0] denorm_final_mant = shifted_denorm[45:23]; // 取高23位
    
    // 最终尾数选择
    wire [FP32_MANT_WIDTH-1:0] computed_mant = result_is_denormal ? denorm_final_mant : final_mant;

    // 确保所有位都有明确的赋值（不会有X值）
    wire [FP32_MANT_WIDTH-1:0] mant_out = 
        output_is_nan    ? {1'b1, {(FP32_MANT_WIDTH-1){1'b0}}} : // NaN (包括inf*0)
        output_is_zero   ? {FP32_MANT_WIDTH{1'b0}} :     // 零
        output_is_inf    ? {FP32_MANT_WIDTH{1'b0}} :     // 无穷大
        exp_overflow     ? {FP32_MANT_WIDTH{1'b0}} :     // 溢出到无穷大
        (exp_underflow && !result_is_denormal) ? {FP32_MANT_WIDTH{1'b0}} : // 下溢到零
        computed_mant;                                   // 正常情况或非规格化数

    // 最终指数值，考虑特殊情况
    wire [FP32_EXP_WIDTH-1:0] exp_out = 
        output_is_nan    ? 8'd255 :                // NaN (包括inf*0)
        output_is_zero   ? 8'd0 :                  // 零
        output_is_inf    ? 8'd255 :                // 无穷大
        exp_overflow     ? 8'd255 :                // 溢出到无穷大
        (exp_underflow && !result_is_denormal) ? 8'd0 : // 下溢到零
        result_is_denormal ? 8'd0 :               // 非规格化数指数为0
        exp_out_biased[7:0];                       // 正常情况
    
    // 输出结果组装
    assign result = {sign_out, exp_out, mant_out};

endmodule

