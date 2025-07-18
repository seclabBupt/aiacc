`timescale 1ns/1ps

module fpadd #(
    parameter PARAM_DR_FPADD_CNT = 4  
)(
    input wire clk,
    input wire rst_n,
    input wire [127:0] dvr_fpadd_s0,
    input wire [127:0] dvr_fpadd_s1,
    output reg [127:0] dr_fpadd_d,
    input wire inst_valid,
    input wire mode_flag,  // 仅使用最低位：0=16bit×8, 1=32bit×4
    output wire idle
);

    localparam STATE_IDLE = 2'b00;
    localparam STATE_PROC = 2'b01;
    localparam STATE_WAIT = 2'b10;
    localparam STATE_DONE = 2'b11;

    reg [1:0] current_state, next_state;
    reg [127:0] src0_reg, src1_reg;
    reg mode_flag_reg;  
    reg [127:0] result_reg;

    wire [127:0] subword_result;
    assign idle = (current_state == STATE_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= STATE_IDLE;
            src0_reg      <= 128'b0;
            src1_reg      <= 128'b0;
            mode_flag_reg <= 1'b0;
            result_reg    <= 128'b0;
            dr_fpadd_d    <= 128'b0;
        end else begin
            current_state <= next_state;
            if (current_state == STATE_IDLE && inst_valid) begin
                src0_reg      <= dvr_fpadd_s0;
                src1_reg      <= dvr_fpadd_s1;
                mode_flag_reg <= mode_flag[0]; 
            end
            if (current_state == STATE_WAIT) begin
                result_reg <= subword_result;
            end
            if (current_state == STATE_DONE) begin
                dr_fpadd_d <= result_reg;
            end
        end
    end

    always @(*) begin
        next_state = current_state;
        case (current_state)
            STATE_IDLE: next_state = inst_valid ? STATE_PROC : STATE_IDLE;
            STATE_PROC: next_state = STATE_WAIT;
            STATE_WAIT: next_state = STATE_DONE;
            STATE_DONE: next_state = STATE_IDLE;
            default:    next_state = STATE_IDLE;
        endcase
    end

    subword_adder u_subword_adder (
        .clk(clk),
        .rst_n(rst_n),
        .src0(src0_reg),
        .src1(src1_reg),
        .mode_flag(mode_flag_reg),  
        .result(subword_result)
    );

endmodule