# C/C++ Programming & Cipher Algorithms

## Overview
A foundational systems programming project featuring robust implementations of classic cryptographic algorithms (like Caesar Cipher) in pure C.

## Technical Details
- **Tech Stack:** C, Systems Programming
- **Core Functionality:** Pointer manipulation, manual memory management, and character array transformations.

## Interesting Concept
The architecture enforces strict separation between algorithmic logic (`cipher.c`) and a comprehensive unit testing framework (`tests.c`), mirroring enterprise-level software engineering practices within a low-level language.

## Key Challenge
**Memory Safety & Buffer Overruns:** Handling raw strings in C often leads to segmentation faults or buffer overflows if null termination characters aren't perfectly managed.
*Solution:* Implemented rigorous bounds checking and robust pointer arithmetics to ensure the cipher routines safely process variable-length inputs without leaking memory or crashing.