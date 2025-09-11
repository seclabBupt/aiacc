//---------------------------------------------------------------------
// Filename: rur.v
// Author: cypher
// Date: 2025-8-24
// Version: 1.1
// Description: This is a module that supports byte-indexing, mask extraction, and result accumulation.
//---------------------------------------------------------------------
module RUR #(
    parameter LOCAL_SMC_ID = 5'd0
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire [96:0]  cru_rur,      // 97 bit 上行指令
    output reg  [127:0] dr_rur_d
);

    // ---------- 97 bit 字段解析 ----------
    wire        vld       = cru_rur[96];
    wire [4:0]  smc_id    = cru_rur[95:91];
    wire [2:0]  ur_id     = cru_rur[90:88];
    wire [7:0]  ur_addr   = cru_rur[87:80];

    // 16 组 {4 bit addr, 1 bit vld}
    wire [3:0]  ur_addr_low      [0:15];
    wire        ur_addr_low_vld  [0:15];

    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : g_field
            assign ur_addr_low     [gi] = cru_rur[79 - gi*5 -: 4]; // 4 bit
            assign ur_addr_low_vld [gi] = cru_rur[79 - gi*5 - 4];  // 1 bit
        end
    endgenerate

    // ---------- UR-RAM ----------
    (* ram_style = "block" *)
    reg [127:0] ur_ram [0:7][0:255];

    // ---------- 主逻辑 ----------
    integer x;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dr_rur_d <= 128'h0;
        end
        else if (vld && (smc_id == LOCAL_SMC_ID)) begin
            automatic logic [127:0] mask = 128'h0;
            automatic logic [127:0] raw  = ur_ram[ur_id][ur_addr];

            for (x = 0; x < 16; x = x + 1) begin
                if (ur_addr_low_vld[x]) begin
                    automatic int sel = ur_addr_low[x];
                    mask[8*x +: 8] = raw[8*sel +: 8];
                end
            end
            dr_rur_d <= dr_rur_d | mask;
        end
    end

endmodule
