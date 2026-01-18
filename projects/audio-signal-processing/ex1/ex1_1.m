% Synthetic WSS Process Generation - AR(1) Model
% Exercise 1.1

clear; clc; close all;

%% 1. Define Parameters
N_samples = 1000;       % Number of samples
alpha = 0.8;           % AR(1) parameter (must be < 1)
sigma_N_sq = 1;        % Variance of additive noise
sigma_N = sqrt(sigma_N_sq);

% Ensure alpha is valid
if abs(alpha) >= 1
    error('Alpha must be less than 1 for WSS process.');
end

%% 2. Generate Innovation Noise {Gn}
% Gn ~ N(0, 1) i.i.d
G = randn(N_samples, 1); 

%% 3. Generate AR(1) Process {Z_tilde}
% Z_tilde_0 ~ N(0, 1/(1-alpha^2))
% Z_tilde_n = alpha * Z_tilde_{n-1} + Gn  for n >= 1

Z_tilde = zeros(N_samples, 1);

% Initialize Z_tilde(1) (which corresponds to n=0 in the math notation if we treat 1 as start, 
% but let's assume we stimulate N samples t=1..N. The problem says n>=1 is the recursion.
% It defines Z_0 separately. Let's create a vector for indices 0 to N-1 or just 1 to N.
% MATLAB is 1-based. Let's align n with indices.
% If we want Z_0, we can store it in a temporary variable or index 1, but usually 
% signal processing vectors are just the stream.
% Let's implement exactly as described:
% We need a sequence of length N. Let's say Z corresponds to n=1, 2, ...
% But the recursion depends on n-1. 
% For n=1, we need Z_{0}.
% So, let's draw Z_0 first.

Z_tilde_prev = randn * sqrt(1 / (1 - alpha^2)); % Z_tilde_0

% We will store the sequence starting from n=1
for n = 1:N_samples
    Z_tilde(n) = alpha * Z_tilde_prev + G(n);
    Z_tilde_prev = Z_tilde(n);
end

%% 4. Generate Additive Noise {Nn}
% Nn ~ N(0, sigma_N^2)
Noise = randn(N_samples, 1) * sigma_N;

%% 5. Construct Observed Process {Zn}
% Zn = Z_tilde_n + Nn
Z = Z_tilde + Noise;

%% 6. Visualization
figure;
subplot(3,1,1);
plot(Z_tilde);
title('AR(1) Process \tilde{Z}_n');
xlabel('n'); ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot(Noise);
title('Additive Gaussian Noise N_n');
xlabel('n'); ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot(Z);
title('Observed Process Z_n = \tilde{Z}_n + N_n');
xlabel('n'); ylabel('Amplitude');
grid on;

% Check variances theoretically vs empirically
fprintf('Theoretical Variance of Z_tilde: %.4f\n', 1/(1-alpha^2));
fprintf('Empirical Variance of Z_tilde:   %.4f\n', var(Z_tilde));
fprintf('Theoretical Variance of Z:       %.4f\n', 1/(1-alpha^2) + sigma_N_sq);
fprintf('Empirical Variance of Z:         %.4f\n', var(Z));
