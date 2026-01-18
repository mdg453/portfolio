% Driver script for Question 3
clear; clc; close all;

% Load data
load('radarreception.mat', 'r1vec', 'r2vec');
load('sigvec.mat', 'sigvec');

% Call the function
% Note: The prompt asks to submit the outputs and plot.
[w1vec, w2vec, xvec, yvec] = radardetect(r1vec, r2vec, sigvec);

% Save the plot for verification
saveas(gcf, 'q3_plot.png');
disp('Calculation complete. Plot saved to q3_plot.png');

% Display first few values to verify
disp('First 5 x values:');
disp(xvec(1:5));
disp('First 5 y values:');
disp(yvec(1:5));
