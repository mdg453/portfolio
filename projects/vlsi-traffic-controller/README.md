# VLSI Traffic Light Controller

## Overview
A comprehensive digital logic project simulating a multi-directional traffic light system. It implements a complex Finite State Machine (FSM) in Verilog to control traffic flows, including safety timing buffers and sensor inputs.

## Technical Details
- **Tech Stack:** Verilog, Digital Logic Design, Icarus Verilog
- **Core Functionality:** FSM design, timing parameterization, and exhaustive simulation verification using testbenches.

## Interesting Concept
The entire traffic logic, which could be dozens of lines of nested `if-else` statements in a software microcontroller, is implemented as a pure hardware state machine. This makes the system mathematically deterministic, instantly responsive, and highly robust against failures that typically plague software environments.

## Key Challenge
**State Explosion & Transition Timing:** Handling the numerous timing requirements (green durations, yellow transitions, all-red safety buffers) risked creating an unmanageable number of unique FSM states if each clock cycle was a state.
*Solution:* Implemented an internal hardware counter (timer module) decoupled from the main FSM. The FSM transitions are triggered by the counter's "timeout" signals rather than dedicating individual states for every clock tick. This drastically reduced the state-space and made the traffic timing easily adjustable via module parameters.
