module add32 (
    input         clk,
    input         rst_n,
    input  [127:0] src0,
    input  [127:0] src1,
    input          sign_s0,
    input          sign_s1,
    output [127:0] dst,
    output reg [127:0] st
);

    genvar i;
    // 组合产生每组的 result -> dst（dst 是 wire）
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_add32
            wire [31:0] a = src0[i*32 +: 32]; // i=0 -> lowest 32 bits
            wire [31:0] b = src1[i*32 +: 32];

            // 符号/无符号扩展（组合）
            wire [63:0] s0_ext = sign_s0 ? {{32{a[31]}}, a} : {32'b0, a};
            wire [63:0] s1_ext = sign_s1 ? {{32{b[31]}}, b} : {32'b0, b};

            // 有符号求和，用于溢出判定；低 32 位为基础结果
            wire signed [63:0] s0_signed = $signed(s0_ext);
            wire signed [63:0] s1_signed = $signed(s1_ext);
            wire signed [63:0] sum_signed = s0_signed + s1_signed;
            wire [31:0] sum_lo = sum_signed[31:0];

            // 溢出/饱和处理（只在任一输入被视为有符号时生效）
            wire [31:0] result = (sign_s0 || sign_s1) ?
                ( ((s0_signed[63] == s1_signed[63]) && (sum_signed[63] != s0_signed[63])) ? 32'h7FFFFFFF : sum_lo )
                : sum_lo;

            assign dst[i*32 +: 32] = result;
        end
    endgenerate

    // 单个组合过程生成 st（唯一驱动源，避免任何 assign st[...]）
    integer j;
    always @(*) begin
        st = 128'b0; // 先清零所有位
        for (j = 0; j < 4; j = j + 1) begin
            reg [31:0] a32;
            reg [31:0] b32;
            reg gt, eq, ls;
            a32 = src0[j*32 +: 32];
            b32 = src1[j*32 +: 32];

            if (sign_s0 || sign_s1) begin
                // 有符号比较（32 位）
                gt = ($signed(a32) > $signed(b32));
                ls = ($signed(a32) < $signed(b32));
                eq = (a32 == b32);
            end else begin
                // 无符号比较
                gt = (a32 > b32);
                ls = (a32 < b32);
                eq = (a32 == b32);
            end

            // 按组写入 3-bit 状态（组0 -> st[2:0], 组1 -> st[34:32], ...）
            st[j*32 +: 3] = {gt, eq, ls};
        end
        // 其它位已被 st = 128'b0 初始化为 0
    end

endmodule
