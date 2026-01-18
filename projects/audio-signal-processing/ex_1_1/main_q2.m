%% Advanced Signal Processing - Assignment 1, Question 2
% Steepest Descent Algorithm (Stationary Case)
%
% 2(a) Spectral Preparation (Eigenvalues)
% 2(b) SD Iterations & Excess MSE
% 2(c) Interpretation (Stability)
% 2(d) Rate Verification vs Theoretical Envelope

clear; close all; clc;

alpha = 0.9; % AR(1) parameter
sigma_N_sq = 0.5; % Innovation variance

L_values = [4, 6]; % L values to test
mu_values = [0.001, 0.01, 0.1, 0.15, 0.2]; % Learning rates
n_iters = 140; % Number of iterations

%% Loop over L
for iL = 1:length(L_values)
    L = L_values(iL);
    fprintf('\n--- Analysis for L = %d ---\n', L);

    % --- 2(a) Spectral Preparation ---
    % Construct R and p analytically for AR(1)
    % r_Z[k] = sigma_N^2 / (1-alpha^2) * alpha^|k|

    R_Z0 = sigma_N_sq / (1 - alpha^2); % R_Z[0]
    r_vec = R_Z0 * alpha.^(0:L-1); % r_vec = [r_0, r_1, ... r_{L-1}]

    R = toeplitz(r_vec);
    p = (R_Z0 * alpha.^(1:L))'; % p_k = E[Z_n Z_{n-k}] = r_Z[k]

    % Optimal Solution
    w_star = R \ p;

    % Eigenvalues
    lambdas = eig(R); % Eigenvalues of R
    lambda_max = max(lambdas); % Maximum eigenvalue
    lambda_min = min(lambdas); % Minimum eigenvalue

    fprintf('Lambda Max: %.4f\n', lambda_max);
    fprintf('Lambda Min: %.4f\n', lambda_min);
    fprintf('Condition Number: %.4f\n', lambda_max/lambda_min);
    fprintf('Stability Bound (2/lambda_max): %.4f\n', 2/lambda_max);

    % --- 2(b) & 2(d) Iterations ---
    fig_mse = figure('Name', sprintf('Q2(b) Excess MSE (L=%d)', L)); % Excess MSE figure
    hold on; grid on; % Hold on for multiple lines, grid on for visibility
    title(sprintf('Relative Excess MSE (L=%d)', L)); % Title
    xlabel('Iteration n'); ylabel('10 log_{10} (E_{ex}(n) / E_{ex}(0)) [dB]'); 
    ylim([-Inf, 20]); % Cap at 20dB

    % Figure for Coeff Norm (2d)
    fig_norm = figure('Name', sprintf('Q2(d) Coeff Norm (L=%d)', L)); % Coeff Norm figure
    hold on; grid on; % Hold on for multiple lines, grid on for visibility
    title(sprintf('Coefficient Error Norm (L=%d)', L)); % Title
    xlabel('Iteration n'); ylabel('10 log_{10} (||c_n||^2 / ||w^*||^2) [dB]'); % Labels

    % Legend entries
    legends_mse = {};
    legends_norm = {};

    colors = lines(length(mu_values));

    for iMu = 1:length(mu_values)
        mu = mu_values(iMu);

        % Initialization
        w = zeros(L, 1);

        % Metrics Storage
        excess_mse_dB = zeros(n_iters, 1);
        coeff_norm_dB = zeros(n_iters, 1);

        % Initial Error
        c0 = w - w_star;
        E_ex_0 = c0' * R * c0;
        norm_w_star_sq = w_star' * w_star;

        % SD Loop
        for n = 1:n_iters % n=1 in loop is iteration 0
            c_n = w - w_star; 

            % Excess MSE
            E_ex_n = c_n' * R * c_n;
            excess_mse_dB(n) = 10 * log10(E_ex_n / E_ex_0);

            % Coeff Norm
            norm_c_sq = c_n' * c_n;
            coeff_norm_dB(n) = 10 * log10(norm_c_sq / norm_w_star_sq);

            % Header for n=0
            if n == n_iters
                break;
            end

            % Update Step
            w = w + mu * (p - R * w);
        end

        % Plot Empirical Curves
        figure(fig_mse);
        plot(0:n_iters-1, excess_mse_dB, 'Color', colors(iMu,:), 'LineWidth', 1.5);

        figure(fig_norm);
        p_emp = plot(0:n_iters-1, coeff_norm_dB, 'Color', colors(iMu,:), 'LineWidth', 1.5);

        legends_mse{end+1} = sprintf('mu = %.3f', mu);
        legends_norm{end+1} = sprintf('mu = %.3f (Emp)', mu);

        % --- 2(d) Theoretical Envelope ---
        rho = max(abs(1 - mu * lambda_min), abs(1 - mu * lambda_max));

        % E_theory(n) = 20 log10(rho^n) + C
        C = coeff_norm_dB(1); % Match C so it starts at the empirical value at n=0
        n_axis = 0:n_iters-1; % n=1 in loop is iteration 0
        E_theory = 20 * log10(rho.^n_axis) + C; % Theoretical envelope

        figure(fig_norm);
        plot(n_axis, E_theory, '--', 'Color', colors(iMu,:), 'LineWidth', 1.0);
        legends_norm{end+1} = sprintf('mu = %.3f (Theory)', mu);

    end

    figure(fig_mse);
    legend(legends_mse, 'Location', 'best');

    figure(fig_norm);
    legend(legends_norm, 'Location', 'best');

end

fprintf('\nDone.\n');
