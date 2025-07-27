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
            // Extract 32-bit chunks from source inputs
            wire [31:0] u0 = src0[i*32 +: 32];
            wire [31:0] u1 = src1[i*32 +: 32];

            // Extend to 33-bit signed values based on sign flags
            wire signed [32:0] s0_val = sign_s0 ? {u0[31], u0} : {1'b0, u0};
            wire signed [32:0] s1_val = sign_s1 ? {u1[31], u1} : {1'b0, u1};

            // Perform signed 33-bit addition
            wire signed [32:0] sum = s0_val + s1_val;

            // Clamp if overflow or underflow occurs
            wire [31:0] sum_clipped =
                ((s0_val > 0) && (s1_val > 0) && (sum < 0))                      ? 32'hFFFFFFFF :
                ((s0_val < 0) && (s1_val < 0) && (sum < -33'sd2147483648))       ? 32'hFFFFFFFF :
                                                                                  sum[31:0];

            // Assign result to corresponding segment of output
            assign dst[i*32 +: 32] = sum_clipped;
        end
    endgenerate

endmodule
