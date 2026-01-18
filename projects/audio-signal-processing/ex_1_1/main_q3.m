%% Advanced Signal Processing - Assignment 1, Question 3
% Least Mean Squares (LMS) Algorithm
% 3(a) Implementation & Experiment Grid
% 3(b) Parameter Trade-off
% 3(c) Tracking time-varying AR(1)

clear; close all; clc;

%% 3(a) Stationary LMS Grid
fprintf('--- 3(a) Stationary LMS Grid ---\n');

% Parameters
Fs = 48000;
T = 10;
N = Fs * T;
alpha = 0.9;
sigma_N_sq = 0.5;

% Generate Signal
% Process Z_n = alpha * Z_{n-1} + N_n
Z = filter(1, [1, -alpha], sqrt(sigma_N_sq) * randn(N, 1));

L_values = 2:6; % L values to test
mu_values = [0.1, 0.01, 0.001, 0.0001]; % Mu values to test

% Pre-calculate Theoretical Optima per L for reference
w_stars = cell(max(L_values), 1);
R0 = sigma_N_sq / (1 - alpha^2);

for L = L_values % Pre-calculate Theoretical Optima per L for reference
    r_vec = R0 * alpha.^(0:L-1); % Construct R vector for Toeplitz [r0, r1, ... rL-1]
    R = toeplitz(r_vec); % Construct Toeplitz matrix
    p = (R0 * alpha.^(1:L))'; % p = [r1, r2, ... rL]^T
    w_stars{L} = R \ p; % Theoretical Optima
end

% Loop over Mu
for iMu = 1:length(mu_values)
    mu = mu_values(iMu);

    fig = figure('Name', sprintf('Q3(a) LMS Coeff Error (mu=%.4f)', mu));
    hold on; grid on;
    title(sprintf('LMS Convergence (mu = %.4f)', mu));
    xlabel('Iteration n'); ylabel('Relative Coeff Error [dB]');

    legends = {};

    for iL = 1:length(L_values)
        L = L_values(iL);
        w_star = w_stars{L};
        norm_w_star_sq = norm(w_star)^2;

        % LMS Initialization
        w = zeros(L, 1);
        coeff_err_dB = zeros(N, 1);

        % Run LMS
        % Let's zero-pad Z to handle edges easily.
        Z_pad = [zeros(L, 1); Z];

        for n = 1:N
            % Regressor vector u = [Z(n-1); ... ; Z(n-L)]
            % In Z_pad, Z(n) is at index n+L. Regressors are Z_pad(n+L-1 : -1 : n).
            u = Z_pad(n+L-1 : -1 : n);
            d = Z(n); % Desired

            y = w' * u; % Prediction
            e = d - y; % Error

            w = w + mu * e * u; % Update

            % Calculate Metric
            c = w - w_star;
            coeff_err_dB(n) = 10 * log10( (c' * c) / norm_w_star_sq );
        end

        % Compute Final NR_dB for Legend
        % NR = 10 log10 ( E[d^2] / E[e^2] )
        % We can approximate using last chunk
        chunk = (N-20000):N;
        % Predict over chunk using final w
        err_sq = 0; sig_sq = 0; % Error and signal power
        err_vec = filter([0; -w], 1, Z) + Z; % e = Z - w^T R = Z - filter(w)
        NR_dB = 10*log10( var(Z(chunk)) / var(err_vec(chunk)) ); % Noise Ratio
        plot(coeff_err_dB, 'LineWidth', 1.2); % Plot Coeff Error
        legends{end+1} = sprintf('L=%d, NR=%.2fdB', L, NR_dB); % Add to legend

    end
    legend(legends, 'Location', 'best');
    ylim([-60, 10]); % Zoom in relevant area
end


%% 3(c) Tracking Time-Varying AR(1)
fprintf('\n--- 3(c) Tracking Time-Varying AR(1) ---\n');

% Parameters
mu_track = [0.01, 0.001, 0.0001];
L_track = [1, 2, 4];
% Time segments
% 0-3s: alpha=0.2
% 3-6s: alpha=0.8
% 6-10s: alpha=-0.5
seg1 = 1 : 3*Fs;
seg2 = (3*Fs + 1) : 6*Fs;
seg3 = (6*Fs + 1) : 10*Fs;

alphas = zeros(N, 1);
alphas(seg1) = 0.2;
alphas(seg2) = 0.8;
alphas(seg3) = -0.5;

% Generate TV AR(1)
% Z_n = alpha[n] * Z_{n-1} + G_n
gn = randn(N, 1); % Unit variance innovation
z_tv = zeros(N, 1);
for k = 2:N
    z_tv(k) = alphas(k) * z_tv(k-1) + gn(k);
end


for iL = 1:length(L_track) % Loop over L
    L = L_track(iL); % L value
    fig_track = figure('Name', sprintf('Q3(c) Tracking (L=%d)', L)); % Figure
    hold on; grid on; % Hold on for multiple lines, grid on for visibility
    title(sprintf('Coefficient Trajectories (L=%d)', L)); 
    ylabel('w_1[n]'); xlabel('Sample n'); 
    colors = lines(length(mu_track));
    legends = {};

    % Prepare padded Z
    z_pad = [zeros(L, 1); z_tv];

    for iMu = 1:length(mu_track)
        mu = mu_track(iMu); % Mu value

        w = zeros(L, 1); % Initialize weights
        w_hist = zeros(L, N); % History of weights

        % Run LMS
        for n = 1:N
            u = z_pad(n+L-1 : -1 : n); % Regressor vector
            d = z_tv(n); % Desired value
            y = w' * u; % Prediction
            e = d - y; % Error
            w = w + mu * e * u; % Update
            w_hist(:, n) = w; % Store history
        end

        % Plot First Coefficient Trajectory w_1[n]
        plot(w_hist(1, :), 'Color', colors(iMu, :));
        legends{end+1} = sprintf('mu=%.4f', mu);

        % Calculate Adaptation Time
        % Detect change points 3s and 6s
        idx_change1 = 3*Fs;
        idx_change2 = 6*Fs;

        % True optimal w1 should follow alpha(t) approximately (for L=1, w*=alpha).
        target1 = 0.8; % Target after 3s
        target2 = -0.5; % Target after 6s

        n_adapt1 = get_adaptation_time(w_hist(1,:), idx_change1, target1, 0.1); % Adaptation time
        n_adapt2 = get_adaptation_time(w_hist(1,:), idx_change2, target2, 0.1); % Adaptation time

        fprintf('L=%d, mu=%.4f -> Adapt Time 1: %.1f ms, Adapt Time 2: %.1f ms\n', ...
            L, mu, n_adapt1/Fs*1000, n_adapt2/Fs*1000);

    end

    % Plot Alphas for reference
    plot(alphas, 'k--', 'LineWidth', 0.5);
    legends{end+1} = 'True Alpha';
    legend(legends);
end

fprintf('\nDone.\n');

function ta = get_adaptation_time(w_traj, start_idx, target_val, tol_percent) % Find first index n > start_idx where |w(n) - target| <= tol_percent * |target|

margin = abs(target_val * tol_percent); % Margin

% Search window (e.g. 1 second after change)
search_range = start_idx : min(start_idx + 48000, length(w_traj));

rem_dist = abs(w_traj(search_range) - target_val); % Remaining distance

% Filter conditions
% We want it to *remain* <= margin? The question says "falls within".
% Standard def: First crossing.

hits = find(rem_dist <= margin, 1, 'first'); % First hit

if isempty(hits)
    ta = NaN;
else
    ta = hits; % in samples
end
end
