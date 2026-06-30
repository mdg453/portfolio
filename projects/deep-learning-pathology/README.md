# Deep Learning Pathology

## Overview
A preliminary hardware and pathology project containing mechanical CAD models (OpenSCAD, DXF) and technical reports aimed at automating physical pathology lab processes.

## Technical Details
- **Tech Stack:** OpenSCAD, CAD, Hardware Engineering
- **Core Functionality:** Parametric 3D modeling and laser-cut / 3D-printable parts generation.

## Interesting Concept
By utilizing OpenSCAD, the mechanical parts are generated via code rather than drawn by hand. This allows the entire physical hardware design to be version-controlled in Git, just like software.

## Key Challenge
**Rapid Iteration on Physical Hardware:** Traditional CAD tools required slow, manual redesigns every time a motor specification or bearing size changed during prototyping.
*Solution:* Built the mechanical assemblies parametrically in code. Changing a single variable (like `bearing_diameter = 8mm`) automatically recalculated and updated the entire assembly, allowing for instant DXF exports and drastically speeding up the prototyping cycle.
