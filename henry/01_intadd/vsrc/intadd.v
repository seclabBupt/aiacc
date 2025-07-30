module intadd (
    input  [127:0] src_reg0,
    input  [127:0] src_reg1,
    input  [127:0] src_reg2,
    input  [1:0]   precision_s0,
    input  [1:0]   precision_s1,
    input  [1:0]   precision_s2,
    input         sign_s0,
    input         sign_s1,
    input         sign_s2,
    input         inst_valid,
    output [127:0] dst_reg0,
    output [127:0] dst_reg1
);

    wire [127:0] add8_dst0, add8_dst1;
    wire [127:0] add32_dst;

    // Instantiate the add8 module (for 4-bit + 8-bit parallel addition)
    add8 add8_unit (
        .src0(src_reg0),
        .src1(src_reg1),
        .src2(src_reg2),
        .sign_s0(sign_s0),
        .sign_s1(sign_s1),
        .sign_s2(sign_s2),
        .dst0(add8_dst0),
        .dst1(add8_dst1)
    );

    // Instantiate the add32 module (for 32-bit + 32-bit parallel addition)
    add32 add32_unit (
        .src0(src_reg0),
        .src1(src_reg1),
        .sign_s0(sign_s0),
        .sign_s1(sign_s1),
        .dst(add32_dst)
    );

    // Output selection logic
    assign dst_reg0 = (inst_valid && precision_s0 == 2'b00 && precision_s1 == 2'b00 && precision_s2 == 2'b00)
                      ? add8_dst0 :
                      (inst_valid && precision_s0 == 2'b11 && precision_s1 == 2'b11)
                      ? add32_dst : 128'd0;

    assign dst_reg1 = (inst_valid && precision_s0 == 2'b00 && precision_s1 == 2'b00 && precision_s2 == 2'b00)
                      ? add8_dst1 : 128'd0;

endmodule
