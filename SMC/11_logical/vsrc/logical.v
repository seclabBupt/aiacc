`timescale 1ns/1ps
module logical_unit(
    input clk,
    input rst_n,
    input logical_vld_i,
    input [3:0] logical_op_i,
    input logical_precision_i,  // 0:16-bit, 1:32-bit
    input shift_dir_i, // 0:左移位, 1:右移位
    input [31:0] logical_src0_i,
    input [31:0] logical_src1_i,
    input [2:0] fpadd_status_i,
    output reg logical_done_o,
    output reg [31:0] logical_dst_o
);

// 操作码定义
localparam
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

// 主逻辑处理
always @(posedge clk or negedge rst_n) begin
    // 内部寄存器声明
    reg [15:0] src16;
    reg [15:0] shifted;
    reg [31:0] rotated_32;
    reg [15:0] rotated_16;
    reg signed [15:0] signed_src;
    reg [4:0] shift32_val;
    reg [3:0] shift16_val;

    if (!rst_n) begin
        logical_dst_o <= 32'b0;
        logical_done_o <= 1'b0;
    end else if (logical_vld_i) begin
        logical_done_o <= 1'b1;

        case (logical_op_i)
            // 基本逻辑操作
            OP_AND: begin
                if (logical_precision_i) // 32-bit
                    logical_dst_o <= logical_src0_i & logical_src1_i;
                else // 16-bit
                    logical_dst_o <= {16'b0, logical_src0_i[15:0] & logical_src1_i[15:0]};
            end
            OP_OR: begin
                if (logical_precision_i)
                    logical_dst_o <= logical_src0_i | logical_src1_i;
                else
                    logical_dst_o <= {16'b0, logical_src0_i[15:0] | logical_src1_i[15:0]};
            end
            OP_XOR: begin
                if (logical_precision_i)
                    logical_dst_o <= logical_src0_i ^ logical_src1_i;
                else
                    logical_dst_o <= {16'b0, logical_src0_i[15:0] ^ logical_src1_i[15:0]};
            end
            OP_NOT: begin
                if (logical_precision_i)
                    logical_dst_o <= ~logical_src0_i;
                else
                    logical_dst_o <= {16'b0, ~logical_src0_i[15:0]};
            end
            OP_COPY: begin
                if (logical_precision_i)
                    logical_dst_o <= logical_src0_i;
                else
                    logical_dst_o <= {16'b0, logical_src0_i[15:0]};
            end

            // 选择器操作
            OP_SELECT_GT: logical_dst_o <= fpadd_status_i[2] ? logical_src0_i : logical_src1_i;
            OP_SELECT_EQ: logical_dst_o <= fpadd_status_i[1] ? logical_src0_i : logical_src1_i;
            OP_SELECT_LS: logical_dst_o <= fpadd_status_i[0] ? logical_src0_i : logical_src1_i;

            // 移位操作重构
            OP_LOGIC_SHIFT: begin
                if (logical_precision_i) begin // 32-bit
                    shift32_val = logical_src1_i[4:0];
                    if (shift_dir_i) // 右移
                        logical_dst_o <= logical_src0_i >> shift32_val; // 逻辑右移
                    else // 左移
                        logical_dst_o <= logical_src0_i << shift32_val;
                end else begin // 16-bit
                    shift16_val = logical_src1_i[3:0];
                    if (shift_dir_i) // 右移
                        shifted = logical_src0_i[15:0] >> shift16_val;
                    else // 左移
                        shifted = logical_src0_i[15:0] << shift16_val;
                    logical_dst_o <= {16'b0, shifted};
                end
            end

            OP_ARITH_SHIFT: begin
                if (logical_precision_i) begin // 32-bit
                    shift32_val = logical_src1_i[4:0];
                    if (shift_dir_i) // 算术右移
                        logical_dst_o <= $signed(logical_src0_i) >>> shift32_val;
                    else // 算术左移
                        logical_dst_o <= $signed(logical_src0_i) <<< shift32_val;
                end else begin // 16-bit
                    shift16_val = logical_src1_i[3:0];
                    signed_src = $signed(logical_src0_i[15:0]);
                    if (shift_dir_i) // 算术右移
                        shifted = signed_src >>> shift16_val;
                    else // 算术左移
                        shifted = signed_src <<< shift16_val;
                    // 16位符号扩展（关键！）
                    logical_dst_o <= shifted[15] ? {16'hFFFF, shifted} : {16'h0, shifted};
                end
            end

            // 旋转移位 - 修复实现
            OP_ROT_SHIFT: begin
                if (logical_precision_i) begin // 32-bit
                    shift32_val = logical_src1_i[4:0] % 32; // 移位量取模32
                    rotated_32 = (logical_src0_i >> shift32_val) | 
                                 (logical_src0_i << (32 - shift32_val));
                    logical_dst_o <= rotated_32;
                end else begin // 16-bit
                    shift16_val = logical_src1_i[3:0] % 16; // 移位量取模16
                    rotated_16 = (logical_src0_i[15:0] >> shift16_val) | 
                                 (logical_src0_i[15:0] << (16 - shift16_val));
                    logical_dst_o <= {16'b0, rotated_16};
                end
            end

            default: logical_dst_o <= 32'b0;
        endcase
    end else begin
        logical_done_o <= 1'b0;
    end
end

endmodule