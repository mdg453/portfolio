%% Verify Question 6 (adaptivepredict.m)
clear; clc;

fprintf('--- Verifying adaptivepredict.m ---\n');

%% 1. Basic Functionality
N = 6000;
z = randn(N, 1);
try
    [z_lms, z_rls] = adaptivepredict(z);
    fprintf('[PASS] Function ran without error.\n');
    if isscalar(z_lms) && isscalar(z_rls)
        fprintf('[PASS] Outputs are scalars.\n');
    else
        fprintf('[FAIL] Outputs are NOT scalars.\n');
    end
catch ME
    fprintf('[FAIL] Error running function: %s\n', ME.message);
end

%% 2. Scale Invariance
% Requirement: f(c * z) = c * f(z)
c = 123.45;
z_scaled = z * c;

[z_lms_s, z_rls_s] = adaptivepredict(z_scaled);

err_lms = abs(z_lms_s - c * z_lms) / (abs(c * z_lms) + 1e-6);
err_rls = abs(z_rls_s - c * z_rls) / (abs(c * z_rls) + 1e-6);

fprintf('Scale Factor c=%.2f\n', c);
fprintf('  LMS Scaled Prediction Error (Rel): %.2e\n', err_lms);
fprintf('  RLS Scaled Prediction Error (Rel): %.2e\n', err_rls);

if err_lms < 1e-4 && err_rls < 1e-4
    fprintf('[PASS] Scale Invariance Verified.\n');
else
    fprintf('[FAIL] Scale Invariance Issue.\n');
end

%% 3. Robustness (Silence/Zeros)
z_silent = zeros(1000, 1);
[z0_lms, z0_rls] = adaptivepredict(z_silent);
if z0_lms == 0 && z0_rls == 0
    fprintf('[PASS] Handled silence (returns 0).\n');
else
    fprintf('[WARN] Silence returned non-zero: LMS=%f, RLS=%f\n', z0_lms, z0_rls);
end

%% 4. Short Input (< 5000, < 32)
z_short = randn(500, 1);
[zs_lms, ~] = adaptivepredict(z_short);
fprintf('[PASS] Handled short input N=500 (Prediction=%.4f)\n', zs_lms);

z_tiny = randn(10, 1);
[zt_lms, zt_rls] = adaptivepredict(z_tiny);
if zt_lms == 0
    fprintf('[PASS] Handled tiny input N=10 < L (returned 0)\n');
else
    fprintf('[FAIL] Tiny input returned %f\n', zt_lms);
end

%% 5. Row Vector Support
z_row = z';
[zr_lms, zr_rls] = adaptivepredict(z_row);
if abs(zr_lms - z_lms) < 1e-9
    fprintf('[PASS] Row vector input supported.\n');
else
    fprintf('[FAIL] Row vector mismatch.\n');
end

fprintf('--- Verification Complete ---\n');
