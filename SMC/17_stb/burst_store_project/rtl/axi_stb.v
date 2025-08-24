module axi_stb (
    input wire clk,
    input wire reset_n,
    input wire [31:0] addr,               // 32-bit address input
    input wire [127:0] data_in,           // 128-bit data input
    input wire [15:0] burst_length,       // Burst length
    input wire write_valid,               // Write valid signal
    output wire axi_awvalid,              // AXI Write Address valid
    output wire axi_wvalid,               // AXI Write valid
    output wire axi_bready,               // AXI Write Response ready
    output wire [31:0] axi_awaddr,        // AXI Write Address
    output wire [127:0] axi_wdata,        // AXI Write Data
    output wire axi_awready,              // AXI Write Address ready
    output wire axi_wready,               // AXI Write Data ready
    input wire axi_bvalid,                // AXI Write Response valid

    // User Register signals
    output wire [11:0] ur_addr,           // User register address
    output wire ur_re,                    // User register read enable
    input wire [127:0] ur_rdata           // User register read data
);

    // Internal variables
    reg [15:0] burst_count;
    reg [31:0] current_addr;
    reg write_in_progress;
    reg [3:0] byte_strobe;                // Byte strobe for burst cycles

    // AXI Address and Data
    assign axi_awvalid = write_valid && !write_in_progress;
    assign axi_wvalid = write_valid && !write_in_progress;
    assign axi_bready = axi_bvalid;
    assign axi_awaddr = current_addr;
    assign axi_wdata = data_in;

    // UR configuration
    assign ur_addr = current_addr[11:0];  // Addressing for UR
    assign ur_re = write_valid && !write_in_progress;

    // Burst handling
    always @(posedge clk or negedge reset_n) begin
        if (~reset_n) begin
            burst_count <= 0;
            current_addr <= 32'b0;
            write_in_progress <= 0;
        end else begin
            if (write_valid && !write_in_progress) begin
                write_in_progress <= 1;
                current_addr <= addr;
                burst_count <= burst_length;
            end else if (axi_awready && axi_wready) begin
                if (burst_count > 0) begin
                    current_addr <= current_addr + 4;  // Increment address by 4
                    burst_count <= burst_count - 1;
                    byte_strobe <= byte_strobe + 1;   // Manage byte strobe for each cycle
                end else begin
                    write_in_progress <= 0;  // Burst complete
                end
            end
        end
    end

    // Byte strobe logic for burst cycle
    always @(posedge clk) begin
        if (write_valid) begin
            case (burst_count)
                1: byte_strobe <= 4'b0001;
                2: byte_strobe <= 4'b0011;
                3: byte_strobe <= 4'b0111;
                4: byte_strobe <= 4'b1111;
                default: byte_strobe <= 4'b1111;
            endcase
        end
    end

endmodule
