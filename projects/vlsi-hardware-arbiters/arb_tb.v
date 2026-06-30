`timescale 1ns/1ps

module arb_tb;

    reg clk;
    reg rst_n;
    reg [2:0] req;
    wire [2:0] gnt;

    // Instantiate the arbiter
    arb uut (
        .clk(clk),
        .rst_n(rst_n),
        .req(req),
        .gnt(gnt)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Function or string for state name
    reg [23:0] state_name;
    always @(*) begin
        case (uut.state)
            3'b000: state_name = "RST";
            3'b001: state_name = "S_0";
            3'b010: state_name = "S_1";
            3'b100: state_name = "S_2";
            default: state_name = "INV";
        endcase
    end

    // Stimulus
    initial begin
        $dumpfile("arb.vcd");
        $dumpvars(0, arb_tb);

        // Initialize inputs
        rst_n = 0;
        req = 3'b000;
        
        // Wait for a few clock cycles before releasing reset
        #12;
        rst_n = 1;

        // Apply specified combinations of requests
        @(negedge clk) req = 3'b000;
        @(negedge clk) req = 3'b001;
        @(negedge clk) req = 3'b010;
        @(negedge clk) req = 3'b100;
        @(negedge clk) req = 3'b111;
        @(negedge clk) req = 3'b110;
        @(negedge clk) req = 3'b100;
        @(negedge clk) req = 3'b000;
        @(negedge clk) req = 3'b111;
        
        // Let it run for a few more cycles to observe behavior
        @(negedge clk);
        @(negedge clk);
        @(negedge clk);
        
        $finish;
    end

    // Clear printouts of grants at each clock cycle
    always @(posedge clk) begin
        // delay by a small amount to allow state and gnt to update
        #1;
        $display("Time=%0t | req=%b | gnt=%b | state=%s", $time, req, gnt, state_name);
    end

endmodule
