`timescale 1ns / 1ps

module intadd_tb;

    reg [127:0] src_reg0;
    reg [127:0] src_reg1;
    reg [127:0] src_reg2;
    reg [1:0]   precision_s0;
    reg [1:0]   precision_s1;
    reg [1:0]   precision_s2;
    reg         sign_s0;
    reg         sign_s1;
    reg         sign_s2;
    reg         inst_valid;

    wire [127:0] dst_reg0;
    wire [127:0] dst_reg1;

    // Instantiate the Unit Under Test (UUT)
    intadd uut (
        .src_reg0(src_reg0),
        .src_reg1(src_reg1),
        .src_reg2(src_reg2),
        .precision_s0(precision_s0),
        .precision_s1(precision_s1),
        .precision_s2(precision_s2),
        .sign_s0(sign_s0),
        .sign_s1(sign_s1),
        .sign_s2(sign_s2),
        .inst_valid(inst_valid),
        .dst_reg0(dst_reg0),
        .dst_reg1(dst_reg1)
    );

    // Task 1: Initialize all inputs
    task init_inputs;
    begin
        src_reg0     = 0;
        src_reg1     = 0;
        src_reg2     = 0;
        precision_s0 = 2'b00;
        precision_s1 = 2'b00;
        precision_s2 = 2'b00;
        sign_s0      = 0;
        sign_s1      = 0;
        sign_s2      = 0;
        inst_valid   = 0;
    end
    endtask

    // Task 2: Random test for 4bit + 8bit mode
    task test_4bit8bit_random;
    integer i;
    begin
        $display("---- 4bit + 8bit RANDOM TEST ----");
        precision_s0 = 2'b00;
        precision_s1 = 2'b00;
        precision_s2 = 2'b00;
        sign_s0 = $random % 2;
        sign_s1 = $random % 2;
        sign_s2 = $random % 2;

        for (i = 0; i < 128; i = i + 4) begin
            src_reg0[i +: 4] = $random;
            src_reg1[i +: 4] = $random;
            src_reg2[i +: 4] = $random;
        end

        inst_valid = 1;
        #5;
        $display("---- 4bit + 8bit RANDOM TEST completed ----\n");
    end
    endtask

    // Task 3: Random test for 32bit + 32bit mode
    task test_32bit_random;
    integer i;
    begin
        $display("---- 32bit + 32bit RANDOM TEST ----");
        precision_s0 = 2'b11;
        precision_s1 = 2'b11;
        precision_s2 = 2'bxx; // unused
        sign_s0 = $random % 2;
        sign_s1 = $random % 2;
        sign_s2 = 0;

        for (i = 0; i < 128; i = i + 32) begin
            src_reg0[i +: 32] = $random;
            src_reg1[i +: 32] = $random;
        end

        src_reg2 = 128'hx;

        inst_valid = 1;
        #5;
        $display("---- 32bit + 32bit RANDOM TEST completed ----\n");
    end
    endtask

    // Task 4: Fully random test (adapted for precision)
    task test_fully_random;
    integer i;
    begin
        $display("---- FULLY RANDOM TEST ----");
        precision_s0 = $random % 4;
        precision_s1 = $random % 4;
        precision_s2 = $random % 4;
        sign_s0 = $random % 2;
        sign_s1 = $random % 2;
        sign_s2 = $random % 2;

        // Default clear
        src_reg0 = 0;
        src_reg1 = 0;
        src_reg2 = 0;

        if (precision_s0 == 2'b11 && precision_s1 == 2'b11) begin
            // 32-bit mode: use 4 chunks of 32-bit
            for (i = 0; i < 128; i = i + 32) begin
                src_reg0[i +: 32] = $random;
                src_reg1[i +: 32] = $random;
            end
        end else if (precision_s0 == 2'b00 && precision_s1 == 2'b00 && precision_s2 == 2'b00) begin
            // 4-bit + 8-bit mode
            for (i = 0; i < 128; i = i + 4) begin
                src_reg0[i +: 4] = $random;
                src_reg1[i +: 4] = $random;
                src_reg2[i +: 4] = $random;
            end
        end

        inst_valid = $random % 2;
        #5;

        // Valid combination check
        if (!((precision_s0 == 2'b00 && precision_s1 == 2'b00 && precision_s2 == 2'b00) ||
              (precision_s0 == 2'b11 && precision_s1 == 2'b11))) begin
            if (dst_reg0 !== 128'd0 || dst_reg1 !== 128'd0)
                $display("Unexpected output for invalid precision combination!");
        end

        $display("---- FULLY RANDOM TEST completed ----\n");
    end
    endtask

    // Task 5: Boundary value test (aligned exactly at boundaries)
    task test_boundary_cases;
    begin
        $display("---- BOUNDARY TEST (Aligned at exact edges) ----");

        // ----------- Case 1: 4-bit + 8-bit unsigned: sum == 255 (no overflow) --------------
        precision_s0 = 2'b00;
        precision_s1 = 2'b00;
        precision_s2 = 2'b00;
        sign_s0 = 0;
        sign_s1 = 0;
        sign_s2 = 0;
        inst_valid = 1;

        // 2 + 254 
        src_reg0 = {124'd0, 4'b0010};  
        src_reg1 = {124'd0, 4'b1110};
        src_reg2 = {124'd0, 4'b1111};
        #5;

        // ----------- Case 2: 4-bit + 8-bit signed: sum == -128 (just at min signed 8-bit) --------------
        sign_s0 = 1;
        sign_s1 = 1;
        sign_s2 = 1;

        // (-4) + (-127)
        src_reg0 = {124'd0, 4'b1100};  
        src_reg1 = {124'd0, 4'b0001};  
        src_reg2 = {124'd0, 4'b1000};  
        #5;

        // ----------- Case 3: 32-bit unsigned: sum == 0xFFFFFFFF (just below overflow) --------------
        precision_s0 = 2'b11;
        precision_s1 = 2'b11;
        sign_s0 = 0;
        sign_s1 = 0;
        inst_valid = 1;

        // 0xFFFFFFFE + 2 
        src_reg0 = {96'd0, 32'hFFFFFFFE};
        src_reg1 = {96'd0, 32'h00000002};
        #5;

        // ----------- Case 4: 32-bit signed: sum == -2147483648 (exact lower bound) --------------
        sign_s0 = 1;
        sign_s1 = 1;

        // -2147483647 + (-2) = -2147483649
        src_reg0 = {96'd0, 32'h80000001}; // -2147483647
        src_reg1 = {96'd0, 32'hFFFFFFFE}; // -2
        #5;

        $display("---- BOUNDARY TEST completed ----\n");
    end
    endtask

	// Initial block to run the tests
	integer k;

	initial begin
		init_inputs; 
		#10;

		// Run 4bit + 8bit random test 10 times
		for (k = 0; k < 10; k = k + 1) begin
			test_4bit8bit_random;
			#10;
		end

		// Run 32bit + 32bit random test 10 times
		for (k = 0; k < 10; k = k + 1) begin
			test_32bit_random;
			#10;
		end

		// Run fully random precision configuration test 10 times
		for (k = 0; k < 100; k = k + 1) begin
			test_fully_random;
			#10;
		end

        // Run boundary tests once
        test_boundary_cases;
        #10;

		$display("All tests completed.");
		$finish;
	end

endmodule
