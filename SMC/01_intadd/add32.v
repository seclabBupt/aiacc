module add32 (
    input  [127:0] src0,       
    input  [127:0] src1,       
    input          sign_s0,    
    input          sign_s1,    
    output [127:0] dst         
);
    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_add32
            wire [31:0] a = src0[i*32 +: 32];
            wire [31:0] b = src1[i*32 +: 32];
            wire signed [63:0] s0 = sign_s0 ? $signed(a) : {32'b0, a};
            wire signed [63:0] s1 = sign_s1 ? $signed(b) : {32'b0, b};
            wire signed [63:0] sum = s0 + s1;
            wire [31:0] result;
            assign result =
                (sign_s0 || sign_s1) ? (
                    // 溢出检测：同号加，结果符号变
                    ((s0[63] == s1[63]) && (sum[63] != s0[63])) ? 32'h7FFFFFFF :
                    sum[31:0]
                ) : sum[31:0];
            assign dst[i*32 +: 32] = result;
        end
    endgenerate
    
endmodule
