`timescale 1ns/1ps

module part2_tb;

    reg clk;
    reg rst_n;
    reg button;
    reg w_5, w_6, x_6, w_7;
    
    wire pulse_rise;
    wire pulse_fall;
    wire z_5;
    wire z_6;
    wire z_7;

    rising_edge_det u_rise (.clk(clk), .button(button), .pulse(pulse_rise));
    falling_edge_det u_fall (.clk(clk), .button(button), .pulse(pulse_fall));
    fsm_w u_fsm_w (.clk(clk), .rst_n(rst_n), .w(w_5), .z(z_5));
    fsm_wx u_fsm_wx (.clk(clk), .rst_n(rst_n), .w(w_6), .x(x_6), .z(z_6));
    fsm_1101 u_fsm_1101 (.clk(clk), .rst_n(rst_n), .w(w_7), .z(z_7));

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("part2.vcd");
        $dumpvars(0, part2_tb);

        rst_n = 0;
        button = 0;
        w_5 = 0; w_6 = 0; x_6 = 0; w_7 = 0;
        #12;
        rst_n = 1;

        $display("--- Testing Edge Detectors ---");
        @(negedge clk) button = 1;
        @(negedge clk) button = 1;
        @(negedge clk) button = 0;
        @(negedge clk) button = 0;

        $display("--- Testing FSM 5 (w=1 for two preceding cycles) ---");
        // w: 0 1 0 1 1 0 1 1 1 0 1
        @(negedge clk) w_5 = 0;
        @(negedge clk) w_5 = 1;
        @(negedge clk) w_5 = 0;
        @(negedge clk) w_5 = 1;
        @(negedge clk) w_5 = 1;
        @(negedge clk) w_5 = 0;
        @(negedge clk) w_5 = 1;
        @(negedge clk) w_5 = 1;
        @(negedge clk) w_5 = 1;
        @(negedge clk) w_5 = 0;
        @(negedge clk) w_5 = 1;

        $display("--- Testing FSM 6 (w!=x for three cycles) ---");
        @(negedge clk) w_6 = 0; x_6 = 1; // 1 cycle
        @(negedge clk) w_6 = 1; x_6 = 0; // 2 cycles
        @(negedge clk) w_6 = 0; x_6 = 1; // 3 cycles -> z=1
        @(negedge clk) w_6 = 1; x_6 = 1; // 0 cycles
        @(negedge clk) w_6 = 0; x_6 = 1; // 1 cycle
        
        $display("--- Testing FSM 7 (1101) ---");
        @(negedge clk) w_7 = 1;
        @(negedge clk) w_7 = 1;
        @(negedge clk) w_7 = 0;
        @(negedge clk) w_7 = 1; // Sequence 1101 complete -> z=1
        @(negedge clk) w_7 = 1; // Sequence is now 11011 (suffix 11)
        @(negedge clk) w_7 = 0; // Suffix 110
        @(negedge clk) w_7 = 1; // Sequence 1101 again -> z=1

        @(negedge clk);
        @(negedge clk);
        $finish;
    end

    always @(posedge clk) begin
        #1;
        $display("Time=%0t | btn=%b rise=%b fall=%b | w5=%b z5=%b | w6=%b x6=%b z6=%b | w7=%b z7=%b",
            $time, button, pulse_rise, pulse_fall, w_5, z_5, w_6, x_6, z_6, w_7, z_7);
    end

endmodule
