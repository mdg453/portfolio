module arb (
    input  wire       clk,
    input  wire       rst_n, // active low and asynchronous
    input  wire [2:0] req,
    output reg  [2:0] gnt
);

    parameter RST = 3'b000;
    parameter S_0 = 3'b001;
    parameter S_1 = 3'b010;
    parameter S_2 = 3'b100;

    reg [2:0] state, next_state;

    // State transition
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RST;
        end else begin
            state <= next_state;
        end
    end

    // Next state logic
    always @(*) begin
        case (state)
            RST: begin
                if (req[0])      next_state = S_0;
                else if (req[1]) next_state = S_1;
                else if (req[2]) next_state = S_2;
                else             next_state = RST;
            end
            S_0: begin
                if (req[1])      next_state = S_1;
                else if (req[2]) next_state = S_2;
                else if (req[0]) next_state = S_0;
                else             next_state = RST;
            end
            S_1: begin
                if (req[2])      next_state = S_2;
                else if (req[0]) next_state = S_0;
                else if (req[1]) next_state = S_1;
                else             next_state = RST;
            end
            S_2: begin
                if (req[0])      next_state = S_0;
                else if (req[1]) next_state = S_1;
                else if (req[2]) next_state = S_2;
                else             next_state = RST;
            end
            default: next_state = RST;
        endcase
    end

    // Output logic
    always @(*) begin
        gnt = state;
    end

endmodule
