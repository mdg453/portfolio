`timescale 1ns / 1ps

module Traffic_lights_tb;

    reg clk;
    reg reset;
    reg S;
    
    wire hr, hy, hg;
    wire cr, cy, cg;
    
    // Instantiate the Unit Under Test (UUT)
    // Scale down the timers for simulation
    // 3 seconds -> 30 clock cycles
    // 20 seconds -> 200 clock cycles
    traffic_lights #(
        .TIMER_3_VAL(32'd30),
        .TIMER_20_VAL(32'd200)
    ) uut (
        .clk(clk),
        .reset(reset),
        .S(S),
        .hr(hr), .hy(hy), .hg(hg),
        .cr(cr), .cy(cy), .cg(cg)
    );
    
    // 50 MHz clock generation (20ns period)
    always #10 clk = ~clk;
    
    // Monitor state changes to console
    always @(uut.state) begin
        case(uut.state)
            3'd0: $display("Time %0t: State changed to HG", $time);
            3'd1: $display("Time %0t: State changed to HY", $time);
            3'd2: $display("Time %0t: State changed to HR", $time);
            3'd3: $display("Time %0t: State changed to CG", $time);
            3'd4: $display("Time %0t: State changed to CY", $time);
            3'd5: $display("Time %0t: State changed to CR", $time);
        endcase
    end
    
    initial begin
        // 1. Show state after reset
        clk = 0;
        reset = 1;
        S = 0;
        
        #105;
        reset = 0; // State is HG. Timer20 starts. (Time: 105ns)
        $display("Time %0t: Reset released", $time);
        
        // 2. What happens if S asserted any time in HG
        // Assert S early before the 20s minimum is complete
        #895; // Time: 1000ns
        $display("Time %0t: Asserting S while in HG", $time);
        S = 1;
        
        // It stays in HG until timer20 finishes at 105 + 4000 = 4105ns.
        // Then transitions to HY.
        // We drop S after it has transitioned to HY.
        #3500; // Time: 4500ns
        $display("Time %0t: De-asserting S", $time);
        S = 0;
        
        // 3. Show transitions according to timers
        // HY finishes at 4105 + 600 = 4705ns -> HR
        // HR finishes at 4705 + 600 = 5305ns -> CG
        // CG starts at 5305ns. Timer20 finishes at 9305ns.
        
        // 4. What happens if S asserted any time in CG
        // Let's assert S at 6000ns (while in CG)
        #1500; // Time: 6000ns
        $display("Time %0t: Asserting S while in CG", $time);
        S = 1;
        // timer3 is now continually reset because there is a car waiting.
        
        // Wait until past the 20s mark (9305ns). Let's wait to 10000ns.
        #4000; // Time: 10000ns
        // State should STILL be CG, because S is 1.
        
        // Drop S at 10000ns (intersection clear)
        $display("Time %0t: De-asserting S (intersection clear)", $time);
        S = 0;
        
        // Now timer3 starts counting 3 seconds (600ns).
        // It finishes at 10600ns.
        // At 10600ns, tc_20 is already 1, tc_3 becomes 1. Transitions to CY.
        
        #3000; // Wait to 13600ns. Should have gone CY -> CR -> HG.
        
        $display("Time %0t: Simulation complete", $time);
        $finish;
    end
    
    // Dump waveforms
    initial begin
        $dumpfile("Traffic_lights_tb.vcd");
        $dumpvars(0, Traffic_lights_tb);
    end

endmodule
