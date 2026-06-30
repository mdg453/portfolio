# Palindrome API

## Overview
A high-performance web service designed to detect palindromes and find the longest palindromic substrings within massive text inputs efficiently.

## Technical Details
- **Tech Stack:** Python, FastAPI, Docker
- **Core Functionality:** RESTful API design, algorithmic optimization, and containerized deployment.

## Interesting Concept
While checking if a string is a palindrome is trivial, finding the *longest* palindromic substring in a massive string is computationally expensive with naive approaches. This API provides an optimized mathematical solution over the web.

## Key Challenge
**Algorithmic Time Complexity:** A naive expanding-center approach results in O(n²) time complexity, which leads to API timeouts on very large strings.
*Solution:* Implemented Manacher's Algorithm, reducing the time complexity to O(n). Combined with FastAPI's asynchronous endpoint handling, this allowed the service to process massive payloads instantly without blocking the event loop.
