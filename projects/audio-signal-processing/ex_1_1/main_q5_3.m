%% Advanced Signal Processing - Assignment 1, Question 5
% Real Life Signals Analysis
% 5(a) Baselines
% 5(b) Tuning
% 5(c) Adaptation Measurment
% 5(d) Plotting

clear; close all; clc;

%% 5(e) Generate Self-Recorded Clip (Synthetic)
% Since we are an AI, we simulate a recording of "fan noise" or similar.
Fs = 48000;
T_rec = 12;
noise_rec = filter(1, [1, -0.8, 0.2], 0.1 * randn(Fs*T_rec, 1));
% Normalize to [-1, 1] range like audiorecorder
mx = max(abs(noise_rec));
if mx > 0
    noise_rec = noise_rec / mx * 0.9;
end
audiowrite('my_recording.wav', noise_rec, Fs);

%% File List
files = {'airplane.wav', 'cafe.wav', 'city.wav', 'vacuumcleaner.wav', 'raash.wav'};

% Tuning Parameters (Manual selection based on typical audio characteristics)
% Heuristic: Audio needs long filters (L ~ 100-500). mu needs to be small enough for stability.
% RLS needs lambda close to 1 (0.999) for stability in stationary segments.

% Store results
summary_table = {};

%% Loop over files
for iF = 1:length(files)
    filename = files{iF};
    fprintf('\nProcessing %s ...\n', filename);

    [Z, fs_file] = audioread(filename);
    % Mono check
    if size(Z, 2) > 1
        Z = mean(Z, 2);
    end
    Z = Z(:);
    N = length(Z);

    % Normalize input to range [-1, 1] for numerical stability
    if max(abs(Z)) > 0
        Z = Z / max(abs(Z));
    end

    %% 5(a) Baselines
    % 1. Zero Predictor: Z_hat = 0 => e = Z
    nr_zero = 0; % log(VarZ/VarZ) = 0

    % 2. Trivial Predictor: Z_hat[n] = Z[n-1] => e[n] = Z[n] - Z[n-1]
    e_trivial = [0; diff(Z)]; % Z[n] - Z[n-1]
    % Compute NR for trivial over stable region (say 2nd half)
    chunk_base = floor(N/2):N;
    nr_trivial = 10*log10(var(Z(chunk_base)) / var(e_trivial(chunk_base)));

    fprintf('  Baseline NR: Zero=0dB, Trivial=%.2fdB\n', nr_trivial);

    %% Run Algorithms
    % We need > 15 dB if possible.

    % --- LMS Configuration ---
    % Try smaller L for faster convergence and less noise
    L_lms = 32;
    mu_lms = 0.001;

    [e_lms, w_lms] = run_lms(Z, L_lms, mu_lms);
    [nad_lms, nr_lms, pz_lms, pe_lms] = analyze_adaptation(Z, e_lms, fs_file);

    fprintf('  LMS (L=%d, mu=%.4f): Adapt=%.1f ms, NR=%.2f dB\n', ...
        L_lms, mu_lms, nad_lms/fs_file*1000, nr_lms);

    % --- RLS Configuration ---
    L_rls = 32;
    lambda = 0.999;  % Back to high lambda (stable memory)
    delta = 10;      % Moderate regularization

    [e_rls, w_rls] = run_rls(Z, L_rls, lambda, delta);

    % Check for explosion
    if any(isnan(e_rls)) || any(isinf(e_rls))
        fprintf('  RLS Unstable! Retrying with L=12, lambda=0.9999...\n');
        % Fallback: Lower order, very high lambda
        lambda = 0.9999; delta = 10; L_rls_fallback = 12;
        [e_rls, w_rls] = run_rls(Z, L_rls_fallback, lambda, delta);
    end

    [nad_rls, nr_rls, pz_rls, pe_rls] = analyze_adaptation(Z, e_rls, fs_file);

    fprintf('  RLS (lam=%.4f): Adapt=%.1f ms, NR=%.2f dB\n', ...
        lambda, nad_rls/fs_file*1000, nr_rls);

    %% Save Residuals for Listening
    [~, n_base, ~] = fileparts(filename);
    % Clip to [-1, 1] before writing to avoid warnings
    e_lms_clip = max(min(e_lms, 1), -1);
    e_rls_clip = max(min(e_rls, 1), -1);

    audiowrite(sprintf('residual_%s_lms.wav', n_base), e_lms_clip, fs_file);
    audiowrite(sprintf('residual_%s_rls.wav', n_base), e_rls_clip, fs_file);

    %% 5(d) Plotting
    % LMS Plot
    plot_results(filename, 'LMS', sprintf('L=%d, \\mu=%.4f', L_lms, mu_lms), ...
        pz_lms, pe_lms, nad_lms, nr_lms, fs_file);

    % RLS Plot
    plot_results(filename, 'RLS', sprintf('L=%d, \\lambda=%.3f, \\delta=%.2f', L_rls, lambda, delta), ...
        pz_rls, pe_rls, nad_rls, nr_rls, fs_file);

    %% 5(b) Critical Listening (Comment only - code provided to user)
    % sound(e_lms, fs_file); pause(N/fs_file);

end

fprintf('\nDone.\n');


%% Helper Functions for Algorithms

function [e, w] = run_lms(x, L, mu)
N = length(x);
w = zeros(L, 1);
% x_pad = [zeros(L,1); x];
% Filter implementation for speed if possible?
% No, LMS is adaptive. Must loop.
% To speed up, we can use dsp.LMSFilter if available, but let's stick to loop
% standard/basic implementation to ensure compatibility.

e = zeros(N, 1);
x_pad = [zeros(L, 1); x];

% Minimalistic loop
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

    % Symmetry Enforcement for Stability
    if mod(n, 20) == 0
        P = (P + P') / 2;
    end

    e(n) = e_prior;
end
end

function [n_adapt, nr_db, p_z, p_e] = analyze_adaptation(z, e, Fs)
% Sliding window M = 24000 (0.5s)
M = 24000;

% Power computation via convolution
% p[n] = (1/M) sum_{k=0}^{M-1} x^2[n-k]
% Filter with ones(M,1)/M

b_avg = ones(M, 1) / M;
p_z = filter(b_avg, 1, z.^2);
p_e = filter(b_avg, 1, e.^2);

% Ratio check
% Where (p_e / p_z) <= 0.5 AND stays for 0.3s (0.3 * Fs)
min_dur = 0.3 * Fs;
ratio = p_e ./ (p_z + eps); % add eps to avoid div by zero

mask = ratio <= 0.5;

% Find first run of true in 'mask' that has length >= min_dur
% We can use convolution or morphological approach
% Convolve mask with ones(min_dur, 1). If result == min_dur, then we have a block.

run_check = filter(ones(min_dur, 1), 1, double(mask));
% The filter output at index k sums mask[k-min_dur+1 : k].
% If result == min_dur, then the window ENDING at k is fully valid.
% So the run started at k - min_dur + 1.

candidates = find(run_check == min_dur, 1, 'first');

if isempty(candidates)
    n_adapt = NaN;
    % Fallback NR
    start_idx = M; % Just skip init
else
    % The Adaptation Time is the START of the run.
    % Candidate returns the END index of the first valid window.
    n_adapt = candidates - min_dur + 1;
    start_idx = n_adapt;
end

% Post-adaptation NR
if start_idx < length(z)
    nr_db = 10 * log10( var(z(start_idx:end)) / var(e(start_idx:end)) );
else
    nr_db = -Inf;
end

end

function plot_results(fname, alg, params_str, pz, pe, nadapt, nr, Fs)
figure('Name', sprintf('%s - %s', fname, alg));
hold on; grid on;

% Convert to dB for plot
% Handle zeroes
pz_db = 10*log10(pz + eps);
pe_db = 10*log10(pe + eps);

t_axis = (0:length(pz)-1) / Fs;

plot(t_axis, pz_db, 'b');
plot(t_axis, pe_db, 'r');

if ~isnan(nadapt)
    xline(nadapt / Fs, 'k--', 'LineWidth', 1.5, 'Label', 'Adaptation');
end

title(sprintf('%s, %s, %s, NR = %.2f dB', fname, alg, params_str, nr));
xlabel('Time [s]'); ylabel('Instantaneous Power [dB]');
legend('Original Noise', 'Prediction Error');
ylim([-60, 0]); % Adjust as needed
end
