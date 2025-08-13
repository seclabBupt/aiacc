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
            wire [3:0] u0 = src0[i*4 +: 4];
            wire [3:0] u1 = src1[i*4 +: 4];
            wire [3:0] u2 = src2[i*4 +: 4];

            // 4bit符号扩展到8bit
            wire signed [7:0] s0 = sign_s0 ? {{4{u0[3]}}, u0} : {4'b0, u0};
            wire signed [7:0] s1 = sign_s1 ? {{4{u1[3]}}, u1} : {4'b0, u1};
            wire signed [7:0] s2 = sign_s2 ? {{4{u2[3]}}, u2} : {4'b0, u2};

            // 三输入加法
            wire signed [8:0] sum = s0 + s1 + s2;

            // 饱和到8bit
            wire signed [7:0] sum_sat =
                (sum > 127)  ? 8'sd127  :
                (sum < -128) ? -8'sd128 :
                               sum[7:0];

            // 输出低4位和高4位
            assign dst0[i*4 +: 4] = sum_sat[3:0];
            assign dst1[i*4 +: 4] = sum_sat[7:4];
        end
    endgenerate

endmodule
