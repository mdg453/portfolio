function [znextLMS, znextRLS] = adaptivepredict(zvec)
% ADAPTIVEPREDICT computes the one-step-ahead prediction for the next sample
% following the input vector zvec, using both LMS and RLS algorithms.
%
% Inputs:
%   zvec - Column vector of past signal samples.
%
% Outputs:
%   znextLMS - Prediction using Normalized LMS (NLMS) algorithm.
%   znextRLS - Prediction using Stabilized RLS algorithm.
%
% Algorithm Settings:
%   L = 32 (Filter order)
%   LMS: Normalized LMS (NLMS) with step size mu = 0.01
%   RLS: Lambda = 0.999, Delta = 10 (with symmetry enforcement)

    %% 1. Pre-processing & Robustness
    zvec = zvec(:); % Ensure column vector
    N = length(zvec);
    L = 32; % Filter order

    % Boundary check
    if N < L
        znextLMS = 0;
        znextRLS = 0;
        return;
    end

    % Scale Invariance (AGC): Normalize input to range [-1, 1]
    scale = max(abs(zvec));
    if scale < 1e-9 
        znextLMS = 0;
        znextRLS = 0;
        return;
    end
    z = zvec / scale;

    %% 2. Algorithm Parameters
    
    % LMS Settings (Normalized LMS)
    mu_nlms = 0.01; 
    eps_nlms = 1e-6; 

    % RLS Settings (Stabilized)
    lambda = 0.999; 
    delta = 10;     

    %% 3. Initialization
    % Warm-up loop over last T=5000 samples
    T = min(5000, N);
    start_n = max(L + 1, N - T + 1);

    w_lms = zeros(L, 1);
    w_rls = zeros(L, 1);
    
    % RLS Initialization
    P = (1/delta) * eye(L);

    %% 4. Warm-up Loop (Training)
    for n = start_n : N
        u = z(n-1 : -1 : n-L);
        d = z(n); 

        % --- LMS Update (NLMS) ---
        y_lms = w_lms' * u;
        e_lms = d - y_lms;
        
        norm_u_sq = u' * u;
        w_lms = w_lms + (mu_nlms / (eps_nlms + norm_u_sq)) * e_lms * u;

        % --- RLS Update ---
        Pu = P * u;
        denom = lambda + u' * Pu;
        k = Pu / denom; 

        y_rls = w_rls' * u;     
        e_rls = d - y_rls;     

        w_rls = w_rls + k * e_rls;
        P = (1/lambda) * (P - k * Pu');

        % Stability Safeguard
        if mod(n, 20) == 0
            P = 0.5 * (P + P');
        end
    end

    %% 5. Final Prediction (One Step Ahead)
    u_next = z(N : -1 : N-L+1);

    % Predict in normalized domain
    z_next_lms_norm = w_lms' * u_next;
    z_next_rls_norm = w_rls' * u_next;

    % De-normalize
    znextLMS = z_next_lms_norm * scale;
    znextRLS = z_next_rls_norm * scale;

end