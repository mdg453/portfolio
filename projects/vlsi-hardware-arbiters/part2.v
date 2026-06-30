module rising_edge_det (
    input wire clk,
    input wire button,
    output wire pulse
);
    reg button_d, button_dd;
    always @(posedge clk) begin
        button_d <= button;
        button_dd <= button_d;
    end
    assign pulse = button_d & ~button_dd;
endmodule

module falling_edge_det (
    input wire clk,
    input wire button,
    output wire pulse
);
    reg button_d, button_dd;
    always @(posedge clk) begin
        button_d <= button;
        button_dd <= button_d;
    end
    assign pulse = ~button_d & button_dd;
endmodule

module fsm_w (
    input wire clk,
    input wire rst_n,
    input wire w,
    output reg z
);
    parameter S0 = 2'b00, S1 = 2'b01, S2 = 2'b10;
    reg [1:0] state, next_state;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S0;
        else state <= next_state;
    end
    always @(*) begin
        case (state)
            S0: next_state = w ? S1 : S0;
            S1: next_state = w ? S2 : S0;
            S2: next_state = w ? S2 : S0;
            default: next_state = S0;
        endcase
    end
    always @(*) z = (state == S2);
endmodule

module fsm_wx (
    input wire clk,
    input wire rst_n,
    input wire w,
    input wire x,
    output reg z
);
    parameter S0 = 2'b00, S1 = 2'b01, S2 = 2'b10, S3 = 2'b11;
    reg [1:0] state, next_state;
    wire diff = (w != x);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S0;
        else state <= next_state;
    end
    always @(*) begin
        case (state)
            S0: next_state = diff ? S1 : S0;
            S1: next_state = diff ? S2 : S0;
            S2: next_state = diff ? S3 : S0;
            S3: next_state = diff ? S3 : S0;
            default: next_state = S0;
        endcase
    end
    always @(*) z = (state == S3);
endmodule

module fsm_1101 (
    input wire clk,
    input wire rst_n,
    input wire w,
    output reg z
);
    parameter S0 = 3'd0, S1 = 3'd1, S2 = 3'd2, S3 = 3'd3, S4 = 3'd4;
    reg [2:0] state, next_state;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S0;
        else state <= next_state;
    end
    always @(*) begin
        case (state)
            S0: next_state = w ? S1 : S0;
            S1: next_state = w ? S2 : S0;
            S2: next_state = w ? S2 : S3;
            S3: next_state = w ? S4 : S0;
            S4: next_state = w ? S2 : S0;
            default: next_state = S0;
        endcase
    end
    always @(*) z = (state == S4);
endmodule
