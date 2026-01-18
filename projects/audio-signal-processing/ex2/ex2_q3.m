% ex2_q3.m - Question 3: An Airport Radar
% Combines the driver script and radardetect function into one file
clear; clc; close all;

%% Load data
load('radarreception.mat', 'r1vec', 'r2vec');
load('sigvec.mat', 'sigvec');

%% Call the radar detection function
[w1vec, w2vec, xvec, yvec] = radardetect(r1vec, r2vec, sigvec);

%% Save the plot
saveas(gcf, 'q3_plot.png');
disp('Calculation complete. Plot saved to q3_plot.png');

%% Display first few values to verify
disp('First 5 x values:');
disp(xvec(1:5));
disp('First 5 y values:');
disp(yvec(1:5));

%% =======================================================================
%  RADARDETECT Function
%  =======================================================================
function [w1vec,w2vec,xvec,yvec] = radardetect(r1vec,r2vec,sigvec)
% RADARDETECT Estimates the delay and location of an airplane.
%
%   [w1vec,w2vec,xvec,yvec] = radardetect(r1vec,r2vec,sigvec)
%
%   Inputs:
%       r1vec  - Received signal at antenna 1 (entire duration).
%       r2vec  - Received signal at antenna 2 (entire duration).
%       sigvec - The transmitted radar pulse signal.
%
%   Outputs:
%       w1vec  - Estimated delay (in samples) for antenna 1 for each pulse.
%       w2vec  - Estimated delay (in samples) for antenna 2 for each pulse.
%       xvec   - Estimated x-coordinate (in meters) of the target.
%       yvec   - Estimated y-coordinate (in meters) of the target.

% Constants
Fs = 5e6;               % Sampling rate [samples/sec]
T = 4.096e-4;           % Pulse repetition interval [sec]
c = 3e8;                % Speed of light [m/s]
Delta = 3840;           % Distance between antennas [m]

samples_per_pulse = round(T * Fs); % Should be 2048

% Ensure inputs are column vectors
r1vec = r1vec(:);
r2vec = r2vec(:);

N = length(r1vec) / samples_per_pulse; % Number of pulses

% Reshape into matrices (samples_per_pulse x N)
R1 = reshape(r1vec, samples_per_pulse, N);
R2 = reshape(r2vec, samples_per_pulse, N);

% Matched Filter Setup - time-reversed signal
h = sigvec(end:-1:1);

% Initialize output vectors
w1vec = zeros(N, 1);
w2vec = zeros(N, 1);

% Apply matched filter to all pulses
Y1 = filter(h, 1, R1);
Y2 = filter(h, 1, R2);

% Find the delay for each pulse (MAP estimator)
% Subtract L from peak location (causal filter correction)
L = length(sigvec);

[~, max_idx1] = max(Y1);
[~, max_idx2] = max(Y2);

w1vec = max_idx1(:) - L;
w2vec = max_idx2(:) - L;

% Geometric Triangulation
% d is aggregated distance (full round-trip path length)
d1 = w1vec * c / Fs;
d2 = w2vec * c / Fs;

% Geometry solutionד 
% A = 1/4 * d1^2
% B = (d2 - 1/2 * d1)^2
% x = (A - B) / (2*Delta) + Delta/2
% y = sqrt(A - x^2)

A = 0.25 * d1.^2;
B = (d2 - 0.5 * d1).^2;

xvec = (A - B) ./ (2 * Delta) + Delta / 2;

% Ensure argument for sqrt is non-negative
sq_arg = A - xvec.^2;
sq_arg(sq_arg < 0) = 0;
yvec = sqrt(sq_arg);

% Plot the estimated path
figure;
plot(xvec, yvec, 'b.-');
xlabel('Position x [m]');
ylabel('Position y [m]');
title('Estimated Airplane Path');
grid on;
axis equal;
end
