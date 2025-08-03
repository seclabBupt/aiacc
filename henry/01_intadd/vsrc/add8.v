module add8 (
    input  [127:0] src0,        
    input  [127:0] src1,        
    input  [127:0] src2,        
    input          sign_s0,     
    input          sign_s1,     
    input          sign_s2,     
    output [127:0] dst0,        
    output [127:0] dst1         
);

    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : gen_add8
            // Extract 4-bit values
            wire [3:0] u0 = src0[i*4 +: 4];
            wire [3:0] u1 = src1[i*4 +: 4];
            wire [3:0] u2 = src2[i*4 +: 4];

            // Concatenate u2 and u1 into 8-bit value
            wire [7:0] concat_val = {u2, u1};

            // Extend src0: signed and unsigned versions
            wire signed [7:0] s0_signed   = {{4{u0[3]}}, u0};
            wire        [7:0] s0_unsigned = {4'b0000, u0};

            // Extend concat_val: signed and unsigned versions
            wire signed [7:0] val_signed   = {concat_val[7], concat_val[6:0]};
            wire        [7:0] val_unsigned = concat_val;

            // Promote to 9-bit signed operands
            wire signed [8:0] s0_val  = sign_s0 ? {s0_signed[7], s0_signed}   : {1'b0, s0_unsigned};
            wire signed [8:0] add_val = sign_s2 ? {val_signed[7], val_signed} : {1'b0, val_unsigned};

            // Perform signed addition
            wire signed [8:0] sum_signed = s0_val + add_val;

            // Clamp result on overflow or underflow
            wire [7:0] sum_clipped =
                ((s0_val > 0) && (add_val > 0) && (sum_signed < 0))             ? 8'hFF :
                ((s0_val < 0) && (add_val < 0) && (sum_signed < -9'sd128))      ? 8'hFF :
                                                                                  sum_signed[7:0];

            // Output lower and upper 4 bits
            assign dst0[i*4 +: 4] = sum_clipped[3:0]; 
            assign dst1[i*4 +: 4] = sum_clipped[7:4];  
        end
    endgenerate

endmodule
