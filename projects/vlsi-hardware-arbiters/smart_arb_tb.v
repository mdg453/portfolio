`timescale 1ns/1ps

module smart_arb_tb;

    reg clk;
    reg rst_n;
    reg [2:0] req;
    wire [2:0] gnt;

    // Instantiate the smart arbiter
    Smart_arb uut (
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

    // Stimulus
    initial begin
        $dumpfile("smart_arb.vcd");
        $dumpvars(0, smart_arb_tb);

        // Initialize inputs
        rst_n = 0;
        req = 3'b000;
        
        // Wait for reset
        #12;
        rst_n = 1;

        // Apply scenarios to test history and fairness
        
        // Test 1: req 0 gets grant twice
        @(negedge clk) req = 3'b001; 
        @(negedge clk) req = 3'b001; 

        // Test 2: all requests asserted. Since 0 just got it twice, it should prefer lower number between 1 and 2, which is 1
        @(negedge clk) req = 3'b111; 
        
        // Test 3: all requests asserted again. Now history is (0, 1), so 2 should win
        @(negedge clk) req = 3'b111; 
        
        // Test 4: all requests asserted again. History is (1, 2), so 0 should win
        @(negedge clk) req = 3'b111; 

        // Test 5: req 1 and 2. History is (2, 0). 1 is not in history, so 1 should win
        @(negedge clk) req = 3'b110; 

        // Test 6: No requests. gnt should be 000 and history unchanged.
        @(negedge clk) req = 3'b000; 

        // Test 7: req 0 and 2. History is still (0, 1) from before req=000? 
        // Wait, after req=110, history became (0, 1). 
        // Let's see what happens here!
        @(negedge clk) req = 3'b101; 

        // Additional cycles to observe the output
        @(negedge clk);
        @(negedge clk);
        
        $finish;
    end

    // Clear printouts of grants and history at each clock cycle
    always @(posedge clk) begin
        // delay by a small amount to allow state and gnt to update
        #1;
        $display("Time=%0t | req=%b | gnt=%b | last_one=%0d | last_two=%0d", 
                 $time, req, gnt, uut.last_one_gnt, uut.last_two_gnt);
    end

endmodule
