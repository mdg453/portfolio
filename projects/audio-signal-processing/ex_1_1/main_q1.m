%% Advanced Signal Processing - Assignment 1, Question 1
%
% This script implements:
% 1.4 Generation of Synthetic WSS Process (AR(1))
% 1.4(e) Empirical Interference Cancellation
% 1.5 Theoretical Linear Prediction

clear; close all; clc;

%% 1.4 Signal Generation
fprintf('--- 1.4 Signal Generation ---\n');

% Parameters
Fs = 48000;         % Sampling Frequency (Hz)
T_dur = 10;         % Duration (seconds)
N = Fs * T_dur;     % Number of samples
alpha = 0.72;       % AR(1) parameter
sigma_N_sq = 0.28;  % Noise variance

% Generate IID Gaussian Noise
N_proc = sqrt(sigma_N_sq) * randn(N, 1);

% Generate AR(1) Process Z_n
% Z_n = alpha * Z_{n-1} + N_n
% Filter syntax: a(1)y(n) = b(1)x(n) - a(2)y(n-1)
% y(n) - alpha*y(n-1) = x(n)  => a = [1, -alpha], b = 1
b_ar = 1;
a_ar = [1, -alpha];
Z = filter(b_ar, a_ar, N_proc);

% 1.4(a) Empirical Moments
emp_mean = mean(Z);
emp_var = var(Z); % or mean(Z.^2) since mean is approx 0
theo_var = sigma_N_sq / (1 - alpha^2);

fprintf('Empirical Mean: %.4f (Expected: 0)\n', emp_mean);
fprintf('Empirical Var : %.4f (Expected: %.4f)\n', emp_var, theo_var);

% 1.4(b) Scaling for Playback
target_var = 0.5;
beta = sqrt(target_var / emp_var);
Z_scaled = beta * Z;

fprintf('Scaling factor beta: %.4f\n', beta);

% 1.4(c) Playback (User can uncomment to listen)
% sound(Z_scaled, Fs);
% pause(T_dur + 1);

%% 1.4(e) Reference Channel Interference Cancellation
fprintf('\n--- 1.4(e) Interference Cancellation (Empirical) ---\n');

% Model: R_n = (h * Z)_n + V_n
h = [0.60, -0.30, 0.15]';
sigma_V_sq = 0.25;
V = sqrt(sigma_V_sq) * randn(N, 1);

% Convolution Z * h (use 'filter' for causal implementation)
% h is FIR, so 'a' coef is 1. 'b' coefs are h.
R_measured = filter(h, 1, Z) + V;

X_target = Z; % Primary signal
L_values = 1:5; % L values to test
fprintf('L\tNR (dB)\n'); % Print header
fprintf('----------------\n'); % Print separator

for i = 1:length(L_values)
    L = L_values(i);

    % Solve Wiener Filter (Empirical)
    % Predict X_target(n) from R_measured(n-1...n-L)
    w = linpred_wiener(X_target, R_measured, L);

    % Reconstruct Estimation Z_hat
    % Since w corresponds to R_{n-1}...R_{n-L}, we filter R with [0; w]
    % b = [0; w] means y(n) = 0*x(n) + w1*x(n-1) + ...
    Z_hat = filter([0; w], 1, R_measured);

    % Residual
    e_n = X_target - Z_hat;

    % Compute Noise Reduction (NR)
    % NR_dB = 10 log10 ( E[Z^2] / E[e^2] )
    % Only compute over valid range to avoid startup transients
    valid_mask = (L+100):N;

    pow_Z = mean(X_target(valid_mask).^2);
    pow_e = mean(e_n(valid_mask).^2);

    NR_dB = 10 * log10(pow_Z / pow_e);

    fprintf('%d\t%.4f\n', L, NR_dB);
end

%% 1.5 Theoretical Linear Prediction
fprintf('\n--- 1.5 Theoretical Linear Prediction ---\n');

% New Parameters
alpha_new = 0.82;
sigma_N_sq_new = 0.35;

% Generate new realization for 1.5(b)
N_proc_new = sqrt(sigma_N_sq_new) * randn(N, 1);
Z_new = filter(1, [1, -alpha_new], N_proc_new);

L_range = 1:7; % L values to test
fprintf('L\tAvg Error Power\tNR (dB)\n'); % Print header
fprintf('------------------------------------\n'); % Print separator

for i = 1:length(L_range)
    L = L_range(i);

    R_Z0 = sigma_N_sq_new / (1 - alpha_new^2); % R_Z[0]

    % Construct R vector for Toeplitz [r0, r1, ... rL-1]
    r_vec = R_Z0 * alpha_new.^(0:L-1);
    R_mat = toeplitz(r_vec);

    p_vec = (R_Z0 * alpha_new.^(1:L))'; % p = [r1, r2, ... rL]^T

    % Theoretical Weights
    w_star = R_mat \ p_vec;

    % 1.5(b) Empirical Verification
    % Predict Z_new using Z_new's past
    Z_hat_static = filter([0; w_star], 1, Z_new);
    e_static = Z_new - Z_hat_static;

    % 1.5(c) Metrics
    start_idx = 1000; % Skip transient
    pow_e_static = mean(e_static(start_idx:end).^2);
    pow_Z_new = mean(Z_new(start_idx:end).^2); 

    NR_dB_static = 10 * log10(pow_Z_new / pow_e_static); % NR_dB_static

    fprintf('%d\t%.4f\t\t%.4f\n', L, pow_e_static, NR_dB_static);
end

fprintf('\nDone.\n');
