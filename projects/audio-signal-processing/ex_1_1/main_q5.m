%% Advanced Signal Processing - Question 5
clear; close all; clc;

%% File List
% Ensure 'my_recording.wav' exists in the folder if you want to process it.
files = {'airplane.wav', 'cafe.wav', 'city.wav', 'vacuumcleaner.wav', 'my_recording.wav'};

%% Main Loop over files
for iF = 1:length(files)
    filename = files{iF};

    % Skip if file doesn't exist
    if ~exist(filename, 'file')
        fprintf('\nSkipping %s (File not found)\n', filename);
        continue;
    end

    fprintf('\nProcessing %s ...\n', filename);

    [Z, fs_file] = audioread(filename);
    % Mono conversion if necessary
    if size(Z, 2) > 1 % if size(Z, 2) > 1
        Z = mean(Z, 2); % Z = mean(Z, 2)
    end
    Z = Z(:); % Z = Z(:)
    N = length(Z); % N = length(Z)

    % Normalize input to range [-1, 1] for numerical stability
    if max(abs(Z)) > 0 % if max(abs(Z)) > 0
        Z = Z / max(abs(Z)); % Z = Z / max(abs(Z))
    end

    %% 5(a) Baselines
    % Trivial Predictor: Z_hat[n] = Z[n-1] => e[n] = Z[n] - Z[n-1]
    e_trivial = [0; diff(Z)]; % e_trivial = [0; diff(Z)]
    % Compute NR for trivial over stable region
    chunk_base = floor(N/2):N; % chunk_base = floor(N/2):N
    nr_trivial = 10*log10(var(Z(chunk_base)) / var(e_trivial(chunk_base))); % nr_trivial = 10 log10 ( var(Z(chunk_base)) / var(e_trivial(chunk_base)) )

    fprintf('  Baseline NR: Zero=0dB, Trivial=%.2fdB\n', nr_trivial);

    %% 5(b) + 5(c) + 5(d): Run Algorithms & Analyze

    % --- LMS Configuration ---
    L_lms = 32;
    mu_lms = 0.001;

    [e_lms, ~] = run_lms(Z, L_lms, mu_lms);
    [nad_lms, nr_lms, pz_lms, pe_lms] = analyze_adaptation(Z, e_lms, fs_file);

    fprintf('  LMS (L=%d, mu=%.4f): Adapt=%.1f ms, NR=%.2f dB\n', ...
        L_lms, mu_lms, nad_lms/fs_file*1000, nr_lms);

    % --- RLS Configuration ---
    % Using stabilized parameters
    L_rls = 32; 
    lambda = 0.999;
    delta = 10;

    [e_rls, ~] = run_rls(Z, L_rls, lambda, delta);

    % Check for instability
    if any(isnan(e_rls)) || any(isinf(e_rls))
        fprintf('  RLS Unstable! Retrying with fallback (L=12)...\n');
        [e_rls, ~] = run_rls(Z, 12, 0.9999, 10);
    end

    [nad_rls, nr_rls, pz_rls, pe_rls] = analyze_adaptation(Z, e_rls, fs_file);

    fprintf('  RLS (lam=%.4f): Adapt=%.1f ms, NR=%.2f dB\n', ...
        lambda, nad_rls/fs_file*1000, nr_rls);

    %% 5(d) Plotting
    % Plot LMS
    plot_results(filename, 'LMS', sprintf('L=%d, \\mu=%.4f', L_lms, mu_lms), ...
        pz_lms, pe_lms, nad_lms, nr_lms);

    % Plot RLS
    plot_results(filename, 'RLS', sprintf('L=%d, \\lambda=%.3f, \\delta=%.1f', L_rls, lambda, delta), ...
        pz_rls, pe_rls, nad_rls, nr_rls);
end

fprintf('\nDone.\n');


%% --- Local Functions ---

function [e, w] = run_lms(x, L, mu)
N = length(x);
w = zeros(L, 1);
e = zeros(N, 1);
x_pad = [zeros(L, 1); x];

for n = 1:N
    u = x_pad(n+L-1 : -1 : n);
    d = x(n);
    y = w' * u;
    e_curr = d - y;
    w = w + mu * e_curr * u;
    e(n) = e_curr;
end
end

function [e, w] = run_rls(x, L, lambda, delta)
N = length(x);
w = zeros(L, 1);
P = (1/delta) * eye(L);
x_pad = [zeros(L, 1); x];
e = zeros(N, 1);

for n = 1:N
    u = x_pad(n+L-1 : -1 : n);
    d = x(n);

    % RLS Update
    Pu = P * u;
    denom = lambda + u' * Pu;
    k = Pu / denom;

    y = w' * u;
    e_prior = d - y;

    w = w + k * e_prior;
    P = (1/lambda) * (P - k * Pu');

    % Symmetry Enforcement
    if mod(n, 20) == 0
        P = (P + P') / 2;
    end

    e(n) = e_prior;
end
end

function [n_adapt, nr_db, p_z, p_e] = analyze_adaptation(z, e, Fs)
M = 24000; % 0.5s window

% Moving Average using Filter
b_avg = ones(M, 1) / M;
p_z = filter(b_avg, 1, z.^2);
p_e = filter(b_avg, 1, e.^2);

% Avoid division by zero
ratio = p_e ./ (p_z + eps);

% Condition: ratio <= 0.5 for at least 0.3s
min_dur = round(0.3 * Fs);
mask = ratio <= 0.5;

run_check = filter(ones(min_dur, 1), 1, double(mask));

idx_end = find(run_check == min_dur, 1, 'first');

if isempty(idx_end)
    n_adapt = NaN;
    start_idx = M + 1;
else
    n_adapt = idx_end - min_dur + 1;
    start_idx = n_adapt;
end

if start_idx < length(z) - 100
    nr_db = 10 * log10( var(z(start_idx:end)) / var(e(start_idx:end)) );
else
    nr_db = -Inf;
end
end

function plot_results(fname, alg, params_str, pz, pe, nadapt, nr)
figure('Name', sprintf('%s - %s', fname, alg));
hold on; grid on;

pz_db = 10*log10(pz + eps);
pe_db = 10*log10(pe + eps);

samples = 1:length(pz);

plot(samples, pz_db, 'b', 'LineWidth', 1);
plot(samples, pe_db, 'r', 'LineWidth', 1);

if ~isnan(nadapt)
    xline(nadapt, 'k--', 'LineWidth', 1.5, 'Label', 'Adaptation');
end

title({sprintf('%s, %s', fname, alg), ...
    sprintf('%s, noise reduction = %.2f dB', params_str, nr)});
xlabel('sample number');
ylabel('instantaneous power [dB]');
legend('original noise power', 'prediction error power');
ylim([-60, 5]);
end