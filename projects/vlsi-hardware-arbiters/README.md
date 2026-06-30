# Hardware Arbiters & State Machines

## Overview
A Front-End VLSI design project implementing various bus arbitration protocols using Verilog. The project features RTL designs for both Round-Robin and Smart (Priority-based) Arbiters to manage shared hardware resources.

## Technical Details
- **Tech Stack:** Verilog, Icarus Verilog, Digital Logic
- **Core Functionality:** Finite State Machines (FSM), sequential/combinational logic design, and hardware testbenches.

## Interesting Concept
Unlike software where tasks are scheduled by an OS, in hardware, multiple peripherals demanding bus access simultaneously must be resolved within a single clock cycle. This project implements hardware-level arbitration that guarantees fairness (Round-Robin) or respects strict priority hierarchies (Smart Arbiter) with zero software overhead.

## Key Challenge
**Race Conditions & Glitching:** Ensuring the state transitions occur cleanly on the active clock edge without introducing setup/hold time violations.
*Solution:* Strictly adhered to synchronous design principles, physically separating sequential state-updating logic (using non-blocking `<=` assignments) from the combinational next-state logic (using blocking `=` assignments). This prevented race conditions and ensured the arbiter logic settled properly before the next clock pulse.
