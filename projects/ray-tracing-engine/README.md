# 3D Ray Tracing Engine

## Overview
A physically-based rendering engine built from scratch to simulate the physical behavior of light, supporting shadows, reflections, and refractions to generate photorealistic images.

## Technical Details
- **Tech Stack:** C++, Computer Graphics math
- **Core Functionality:** Ray-geometry intersection math, recursive ray tracing, and material BRDFs (Bidirectional Reflectance Distribution Functions).

## Interesting Concept
The engine simulates actual photons bouncing around a mathematical scene, applying Snell's law for refraction and Fresnel equations for realistic glass and mirror materials.

## Key Challenge
**Performance Bottlenecks:** Naively testing every ray against every triangle in complex 3D meshes resulted in rendering times of several hours per frame.
*Solution:* Designed and implemented a Bounding Volume Hierarchy (BVH) spatial acceleration structure. This reduced the time complexity of intersection testing from O(n) to O(log n), decreasing render times by orders of magnitude.
