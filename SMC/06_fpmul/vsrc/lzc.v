// ==============================================================================
// 通用前导零计数器模块 (Generic Leading Zero Counter)
// 功能：计算输入数据的前导零数量
// ==============================================================================

module leading_zero_counter #(
    parameter DATA_WIDTH = 22,
    parameter COUNT_WIDTH = 5
)(
    input  wire [DATA_WIDTH-1:0] data_in,
    output reg  [COUNT_WIDTH-1:0] leading_zeros
);

    integer i;
    
    always @(*) begin
        leading_zeros = DATA_WIDTH;  // 默认全零情况
        for (i = DATA_WIDTH-1; i >= 0; i = i - 1) begin
            if (data_in[i]) begin
                leading_zeros = DATA_WIDTH - 1 - i;
                break;
            end
        end
    end

endmodule
