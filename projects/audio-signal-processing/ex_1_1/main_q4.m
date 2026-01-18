%% Advanced Signal Processing - Assignment 1, Question 4
% Recursive Least Squares (RLS) Algorithm
% 4(a) Implementation & Initialization (delta)
% 4(b) Long-run & Short-iteration zoom plots
% 4(c) NR_dB reporting
% 4(d) Early-iteration transient analysis

clear; close all; clc;

%% Parameters
Fs = 48000; % Sampling frequency
T = 10; % Duration
N = Fs * T; % Number of samples

% Process Parameters
alpha = 0.9; % AR(1) parameter
sigma_N_sq = 0.5; % Innovation variance

% RLS Parameters
L = 2; % Order
lambda = 0.99; % Forgetting factor
delta_values = [0.001, 0.01, 0.1, 1, 10, 100]; % Initialization parameters

% Theoretical Optima (for L=2)
R0 = sigma_N_sq / (1 - alpha^2); % R_Z[0]
r_vec = R0 * alpha.^(0:L-1); % r_vec = [r_0, r_1, ... r_{L-1}]
R_theo = toeplitz(r_vec); % R_Z[k]
p_theo = (R0 * alpha.^(1:L))'; % p_k = E[Z_n Z_{n-k}]
w_star = R_theo \ p_theo; % w^*
norm_w_star_sq = norm(w_star)^2; % ||w^*||^2

%% Signal Generation
% Z_n = alpha * Z_{n-1} + N_n
Z = filter(1, [1, -alpha], sqrt(sigma_N_sq) * randn(N, 1)); % Z_n = alpha * Z_{n-1} + N_n
Z_pad = [zeros(L, 1); Z]; % Z_pad = [Z_0, Z_1, ... Z_{N-1}]

%% RLS Experiment loop
% Store results for plotting
results = struct();

for iD = 1:length(delta_values)
    delta = delta_values(iD);

    % RLS Initialization
    w = zeros(L, 1);
    % P0 = delta^-1 * I
    P = (1/delta) * eye(L);

    % Storage
    coeff_err_dB = zeros(N, 1);

    % RLS Loop
    for n = 1:N
        % Regressor u(n) = [Z(n-1); Z(n-2)]
        u = Z_pad(n+L-1 : -1 : n);
        d = Z(n);

        % Gain vector k
        % P * u can be computed first
        Pu = P * u; % P * u
        denom = lambda + u' * Pu; % lambda + u^T P u
        k = Pu / denom; % k = (P u) / (lambda + u^T P u)

        % A priori error
        y = w' * u; % y = w^T u
        e_prior = d - y; % e_prior = d - y

        % Update Weights
        w = w + k * e_prior; % w = w + k * e_prior

        % P_new = lambda^-1 * (P - k * u^T * P)
        % Note: k = (P u) / (lambda + u^T P u).
        % In standard form: P(n) = lambda^-1 * P(n-1) - lambda^-1 k(n) u^T P(n-1)
        % Using u^T P = (P u)^T since P is symmetric usually.
        P = (1/lambda) * (P - k * Pu'); % P(n) = lambda^-1 * P(n-1) - lambda^-1 k(n) u^T P(n-1)

        % Metric
        c = w - w_star; % c = w - w^*
        coeff_err_dB(n) = 10 * log10( (c'*c) / norm_w_star_sq ); % coeff_err_dB(n) = 10 log10 ( ||c||^2 / ||w^*||^2 )
    end

    % Store Results
    results(iD).delta = delta; % delta
    results(iD).coeff_err_dB = coeff_err_dB; % coeff_err_dB
    results(iD).w_final = w; % w_final

    % Compute Transient Point (-20 dB)
    idx_20dB = find(coeff_err_dB < -20, 1, 'first'); % idx_20dB
    results(iD).idx_20dB = idx_20dB; 
end

%% Compute NR_dB for the "best" setting (usually expected to be similar for RLS)
% Pick middle delta = 0.1 for reporting NR
best_idx = 3;
w_best = results(best_idx).w_final;
% Compute residual over stable portion (last 50%)
chunk = floor(N/2):N; % chunk = floor(N/2):N
err_vec = filter([0; -w_best], 1, Z) + Z; % err_vec = filter([0; -w_best], 1, Z) + Z
NR_dB_val = 10*log10( var(Z(chunk)) / var(err_vec(chunk)) ); % NR_dB_val = 10 log10 ( var(Z(chunk)) / var(err_vec(chunk)) )

%% Plots

% Figure 1: Long Run
fig_long = figure('Name', 'Q4(b) RLS Long Run');
hold on; grid on;
title(sprintf('RLS Convergence (Long Run), Final NR \\approx %.2f dB', NR_dB_val));
xlabel('Iteration n'); ylabel('Relative Coeff Error [dB]');
ylim([-80, 20]);

legends = {};
colors = lines(length(delta_values));

for iD = 1:length(delta_values)
    plot(results(iD).coeff_err_dB, 'Color', colors(iD,:), 'LineWidth', 1.0);
    legends{end+1} = sprintf('delta = %.3g', results(iD).delta);
end
legend(legends, 'Location', 'best');


% Figure 2: Short Zoom (First 450)
fig_short = figure('Name', 'Q4(b) RLS Short Zoom'); % fig_short = figure('Name', 'Q4(b) RLS Short Zoom')
hold on; grid on; 
title('RLS Convergence (Short Zoom)');
xlabel('Iteration n'); ylabel('Relative Coeff Error [dB]');
xlim([0, 450]); ylim([-40, 20]);

% Replot for Zoom
for iD = 1:length(delta_values)
    plot(results(iD).coeff_err_dB, 'Color', colors(iD,:), 'LineWidth', 1.5);
end
legend(legends, 'Location', 'best');


%% 4(d) Transient Analysis Report
fprintf('\n--- 4(d) Transient Analysis (Reaching -20dB) ---\n');
for iD = 1:length(delta_values)
    idx = results(iD).idx_20dB;
    if isempty(idx)
        fprintf('delta = %.3g: Not reached\n', results(iD).delta); % fprintf('delta = %.3g: Not reached\n', results(iD).delta)
    else
        fprintf('delta = %.3g: Reached -20dB at n = %d\n', results(iD).delta, idx); % fprintf('delta = %.3g: Reached -20dB at n = %d\n', results(iD).delta, idx)
    end
end

fprintf('\nDone.\n');
