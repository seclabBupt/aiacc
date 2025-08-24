module axi_stb_s (
    input  wire         aclk,
    input  wire         aresetn,
    input  wire [31:0]  s_awaddr,
    input  wire         s_awvalid,
    output wire         s_awready,
    input  wire [127:0] s_wdata,
    input  wire [15:0]  s_wstrb,
    input  wire         s_wvalid,
    output wire         s_wready,
    output reg  [1:0]   s_bresp,
    output reg          s_bvalid,
    input  wire         s_bready,
    input  wire [31:0]  s_araddr,
    input  wire         s_arvalid,
    output wire         s_arready,
    output reg  [127:0] s_rdata,
    output reg  [1:0]   s_rresp,
    output reg          s_rvalid,
    input  wire         s_rready,
    // ur 接口
    output reg  [10:0]  ur_addr,
    output reg          ur_re,
    input  wire [127:0] ur_rdata
);

    assign s_awready = 1'b1;
    assign s_wready  = 1'b1;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_bvalid <= 1'b0;
            s_bresp  <= 2'b00;
        end else begin
            if (s_awvalid && s_wvalid && !s_bvalid) begin
                s_bvalid <= 1'b1;
                s_bresp  <= 2'b00;
            end else if (s_bvalid && s_bready) begin
                s_bvalid <= 1'b0;
            end
        end
    end

    assign s_arready = 1'b1;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            ur_addr <= 11'd0;
            ur_re   <= 1'b0;
            s_rvalid<= 1'b0;
            s_rdata <= 128'd0;
            s_rresp <= 2'b00;
        end else begin
            if (s_arvalid && !s_rvalid) begin
                ur_addr <= s_araddr[12:2];
                ur_re   <= 1'b1;
                s_rdata <= ur_rdata;
                s_rresp <= 2'b00;
                s_rvalid<= 1'b1;
            end else begin
                ur_re <= 1'b0;
                if (s_rvalid && s_rready)
                    s_rvalid <= 1'b0;
            end
        end
    end

endmodule