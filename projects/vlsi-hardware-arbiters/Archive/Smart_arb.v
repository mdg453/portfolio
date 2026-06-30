module Smart_arb (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [2:0] req,
    output reg  [2:0] gnt
);

    reg [1:0] last_one_gnt;
    reg [1:0] last_two_gnt;

    reg [1:0] score0, score1, score2;
    reg [1:0] winner;

    // Calculate scores based on history
    // 2 = not in history (highest priority)
    // 1 = in last_two_gnt but not last_one_gnt
    // 0 = in last_one_gnt (lowest priority)
    always @(*) begin
        score0 = (last_one_gnt == 2'd0) ? 2'd0 : ((last_two_gnt == 2'd0) ? 2'd1 : 2'd2);
        score1 = (last_one_gnt == 2'd1) ? 2'd0 : ((last_two_gnt == 2'd1) ? 2'd1 : 2'd2);
        score2 = (last_one_gnt == 2'd2) ? 2'd0 : ((last_two_gnt == 2'd2) ? 2'd1 : 2'd2);
    end

    // Determine the winner among active requesters
    always @(*) begin
        case (req)
            3'b000: winner = 2'd3; // No request
            3'b001: winner = 2'd0; // Only req[0]
            3'b010: winner = 2'd1; // Only req[1]
            3'b100: winner = 2'd2; // Only req[2]
            3'b011: begin // req[0] and req[1]
                if (score0 >= score1) winner = 2'd0;
                else                  winner = 2'd1;
            end
            3'b101: begin // req[0] and req[2]
                if (score0 >= score2) winner = 2'd0;
                else                  winner = 2'd2;
            end
            3'b110: begin // req[1] and req[2]
                if (score1 >= score2) winner = 2'd1;
                else                  winner = 2'd2;
            end
            3'b111: begin // All requests
                if (score0 >= score1 && score0 >= score2) winner = 2'd0;
                else if (score1 > score0 && score1 >= score2) winner = 2'd1;
                else winner = 2'd2;
            end
            default: winner = 2'd3;
        endcase
    end

    // Update state and output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // At reset, prefer lower number requester if no conflicts
            // So we set history such that 0 > 1 > 2
            last_one_gnt <= 2'd2;
            last_two_gnt <= 2'd1;
            gnt          <= 3'b000;
        end else begin
            if (winner != 2'd3) begin
                last_two_gnt <= last_one_gnt;
                last_one_gnt <= winner;
                
                if (winner == 2'd0)      gnt <= 3'b001;
                else if (winner == 2'd1) gnt <= 3'b010;
                else if (winner == 2'd2) gnt <= 3'b100;
            end else begin
                gnt <= 3'b000;
                // last_one_gnt and last_two_gnt remain unchanged when no requests are given
            end
        end
    end

endmodule
