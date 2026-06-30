`timescale 1ns / 1ps

module traffic_lights #(
    // Default values for 50MHz clock. 
    // using Parameter to allow scaling down during simulation.
    parameter TIMER_3_VAL = 32'd150_000_000,
    parameter TIMER_20_VAL = 32'd1_000_000_000
)(
    input clk,
    input reset,
    input S,
    output reg hr, hy, hg,
    output reg cr, cy, cg
);
    localparam HG = 3'd0;
    localparam HY = 3'd1;
    localparam HR = 3'd2;
    localparam CG = 3'd3;
    localparam CY = 3'd4;
    localparam CR = 3'd5;

    reg [2:0] state, next_state;
    
    // Timer registers as requested by the hint
    reg [31:0] timer3;
    reg [31:0] timer20;
    
    wire tc_3 = (timer3 == 0);
    wire tc_20 = (timer20 == 0);
    
    reg timer3_enable, timer3_reset;
    reg timer20_enable, timer20_reset;
    
    
    // 1. State Register (sequential)
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= HG;
        end else begin
            state <= next_state;
        end
    end
    

    // 2. Next State and Output Logic (combinational)
    always @(*) begin
        // Default values to prevent latches
        next_state = state;
        hr = 0; hy = 0; hg = 0;
        cr = 0; cy = 0; cg = 0;
        
        timer3_enable = 0;
        timer20_enable = 0;
        
        case(state)
            HG: begin
                hg = 1; cr = 1;
                timer20_enable = 1;
                timer3_enable = S; 
                if (tc_20 && tc_3) begin
                    next_state = HY;
                end
            end
            HY: begin
                hy = 1; cr = 1;
                timer3_enable = 1;
                if (tc_3) begin
                    next_state = HR;
                end
            end
            HR: begin
                hr = 1; cr = 1;
                timer3_enable = 1;
                if (tc_3) begin
                    next_state = CG;
                end
            end
            CG: begin
                hr = 1; cg = 1;
                timer20_enable = 1;
                timer3_enable = ~S;
                if (tc_20 && tc_3) begin
                    next_state = CY;
                end
            end
            CY: begin
                hr = 1; cy = 1;
                timer3_enable = 1;
                if (tc_3) begin
                    next_state = CR;
                end
            end
            CR: begin
                hr = 1; cr = 1;
                timer3_enable = 1;
                if (tc_3) begin
                    next_state = HG;
                end
            end
            default: next_state = HG;
        endcase
    end


    // 3. Timer Reset Logic (combinational)
    always @(*) begin
        // By default, reset timers when transitioning to a new state
        timer3_reset = (state != next_state);
        timer20_reset = (state != next_state);
        
        // Specific reset conditions inside states for the sensor
        if (state == HG && !S) begin
            timer3_reset = 1'b1; // Restart 3s wait if S drops
        end
        if (state == CG && S) begin
            timer3_reset = 1'b1; // Restart 3s wait if S becomes active again
        end
    end
    

    // 4. Timer 3 Implementation (Sequential)

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            timer3 <= TIMER_3_VAL;
        end else if (timer3_reset) begin
            timer3 <= TIMER_3_VAL;
        end else if (timer3_enable && !tc_3) begin
            timer3 <= timer3 - 1;
        end
    end
    

    // 5. Timer 20 Implementation (Sequential)
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            timer20 <= TIMER_20_VAL;
        end else if (timer20_reset) begin
            timer20 <= TIMER_20_VAL;
        end else if (timer20_enable && !tc_20) begin
            timer20 <= timer20 - 1;
        end
    end

endmodule
