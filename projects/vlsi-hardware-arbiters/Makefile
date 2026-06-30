all: arb smart_arb part2

arb: arb.v arb_tb.v
	iverilog -o arb.vvp arb.v arb_tb.v
	vvp arb.vvp

smart_arb: Smart_arb.v smart_arb_tb.v
	iverilog -o smart_arb.vvp Smart_arb.v smart_arb_tb.v
	vvp smart_arb.vvp

part2: part2.v part2_tb.v
	iverilog -o part2.vvp part2.v part2_tb.v
	vvp part2.vvp

clean:
	rm -f *.vvp *.vcd
