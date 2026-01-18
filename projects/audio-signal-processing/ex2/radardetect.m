function [w1vec,w2vec,xvec,yvec] = radardetect(r1vec,r2vec,sigvec)
% RADARDETECT Estimates the delay and location of an airplane.
%
%   [w1vec,w2vec,xvec,yvec] = radardetect(r1vec,r2vec,sigvec)
%
%   Inputs:
%       r1vec  - Received signal at antenna 1 (entire duration).
%       r2vec  - Received signal at antenna 2 (entire duration).
%       sigvec - The transmitted radar pulse signal.
%
%   Outputs:
%       w1vec  - Estimated delay (in samples) for antenna 1 for each pulse.
%       w2vec  - Estimated delay (in samples) for antenna 2 for each pulse.
%       xvec   - Estimated x-coordinate (in meters) of the target.
%       yvec   - Estimated y-coordinate (in meters) of the target.

% Constants
Fs = 5e6;               % Sampling rate [samples/sec]
T = 4.096e-4;           % Pulse repitition interval [sec]
c = 3e8;                % Speed of light [m/s]
Delta = 3840;           % Distance between antennas [m]

samples_per_pulse = round(T * Fs); % Should be 2048

% Ensure inputs are column vectors
r1vec = r1vec(:);
r2vec = r2vec(:);

% Verify lengths are multiples of samples_per_pulse
if mod(length(r1vec), samples_per_pulse) ~= 0
    error('Input vector length must be a multiple of T*Fs');
end

N = length(r1vec) / samples_per_pulse; % Number of pulses

% Reshape into matrices (samples_per_pulse x N)
R1 = reshape(r1vec, samples_per_pulse, N);
R2 = reshape(r2vec, samples_per_pulse, N);

% Matched Filter Setup
% The matched filter is the time-reversed signal.
% We use 'filter' which works along columns by default.
h = sigvec(end:-1:1);

% Initialize output vectors
w1vec = zeros(N, 1);
w2vec = zeros(N, 1);

% Process each pulse
% Note: While we could vectorize the filter operation on the entire matrix,
% iterating is often clearer for extraction of max indices per column without
% complex matrix manipulation, though filter(h,1,R1) works too. Let's use
% the matrix version for efficiency.

Y1 = filter(h, 1, R1);
Y2 = filter(h, 1, R2);

% Find the delay for each pulse (MAP estimator)
% The peak of the matched filter occurs at delay + length(sigvec).
% Since causal matched filter has length L, output peak is shifted by L.
% The theoretical non-causal matched filter peaks at 'delay'.
% The prompt says: "Matlab implementation ... time indexes span from 1 to L...
% determine actual delay, we should subtract L from location of maximal matched filter output."
% Actually, standard 'filter' operation: y[n] = sum x[k]h[n-k].
% If h is reversed sig, h[k] = s[L-1-k].
% Peak is at correct delay + L - 1 usually? Let's follow prompt instructions strictly if given,
% or standard theory.
% RECHECK IMAGES: "Therefore, in order to determine the actual delay, we should
% subtract L from the location of the maximal matched filter output."

L = length(sigvec);

[~, max_idx1] = max(Y1); % max_idx1 is a row vector 1xN
[~, max_idx2] = max(Y2);

w1vec = max_idx1(:) - L;
w2vec = max_idx2(:) - L;

% Geometric Triangulation
% Convert delays to distances
% d = w * c / (2 * f) ? No, prompt equation (2) says d = w*c / (2*f) where f is frequency??
% WAIT. Equation (2) in provided image: d = wc / 2f.
% "w is a sample number and f is the frequency".
% Let's look closer at text. "f is the frequency".
% Earlier text says "Fs = 5x10^6 samples per second".
% If w is sample count, Time_delay = w / Fs.
% Distance = Time_delay * c (round trip) = w/Fs * c.
% Aggregated distance (there and back) is usually 2 * R.
% Prompt says "aggregated distance d ... d = w*c/2f" ? No, that looks like "f" might be Fs?
% Let's re-read carefully.
% Image 2 Eq (2): "d = wc/2f"
% "w is a sample number and f is the frequency"
% This is slightly confusing. Usually Distance_One_Way = (Time_Delay * c) / 2.
% Time_Delay = w / Fs.
% So Distance_One_Way = (w * c) / (2 * Fs).
% Aggregated Distance (d_agg) usually means accumulated path length (Transmitter -> Target -> Receiver).
% Text says: "relationship between propagation delay w and aggregated distance d is d = wc/2f".
% IF d is the TOTAL parth length (d1 = R_tx + R_rx), then d_agg = c * Time_Delay = c * (w/Fs).
% This doesn't match d = w*c/(2f) unless "d" in eq(2) is ONE-WAY distance and "aggregated" is a misleading term OR "f" is something else?
% Ah, wait. If "f" is SAMPLING FREQUENCY Fs.
% Then d = w * c / (2 * Fs) implies d corresponds to ONE-WAY distance if it was monostatic.
% Let's look at Eq (4): 2 * sqrt(...) = d1.
% The LHS is 2 * Distance_to_target.
% So d1 in Eq (4) MUST be the ROUND TRIP distance (or sum of distances).
% If LHS is 2*R, then R = d1/2.
% So d1 is indeed the "Aggregated Distance" (Tx -> Target -> Rx).
% In monostatic (colocated), Tx->Tgt->Rx = 2*R.
% So d1 = 2*R = c * (w/Fs).
% Why the factor of 2 in denominator of (2)? "d = wc/2f".
% If d = wc/2Fs, then d1 = 2*R implies 2*R = wc/2Fs => R = wc/4Fs. THIS SEEMS WRONG.
% Standard Radar: R = c * t / 2 = c * (w/Fs) / 2 = wc / 2Fs.
% So "d" in Eq(2) seems to be the ONE-WAY distance (Range R), NOT the aggregated distance?
% -> Let's check text again carefully.
% "d is in meters, and w is a sample number and f is the frequency."
% "The delay measured by the first antenna is w1, and gives the aggregated distance d1 = w1*c" in TEXT below Figure 3?
% WAIT. Text below img says: "aggregated distance d1 = w1 * c" (Wait, purely w1*c? No division by Fs?)
% Ah, w1 is delay in SECONDS there perhaps? No, "w is a sample number".
% Be careful. The text contains conflicting or specific definitions.
% Let's check the scanned PDF text in the prompt I extracted earlier for "Question 3".
% It says: "The delay measured by the first antenna is w1, and gives the aggregated distance d1 = w1 * c".
% IF w1 is samples, this is physically dimensionally wrong unless w1 is seconds.
% BUT w1vec is requested in samples.
% Let's assume standard physics and adjust to the formulas in text.
% Time_delay = w_samples / Fs.
% Aggregated Distance (Path Length) = c * Time_delay = c * w_samples / Fs.
% Let's verify Eq (6): (x-x1)^2 + (y-y1)^2 = 1/4 * d1^2.
% This is Equation of circle with radius R_1.
% R_1^2 = 1/4 * d1^2 => R_1 = d1 / 2.
% This confirms d1 is the TOTAL ROUND TRIP PATH LENGTH (2 * Range).
% So d1 = c * (w_samples / Fs).
% Now back to Eq (2): "d = wc/2f". If d is defined as one-way distance here?
% If d = R, then R = wc/2Fs. Matches standard radar.
% BUT Eq (6) uses d1 as 2*R.
% So if Eq(2) gives d=R, then d1 in Eq(6) would be 2*d (from Eq 2).
% Let's stick to the Definitions in the Geometric Text part which leads to the solution:
% "d1 = w1 * c" (likely typo in text, meant w1_seconds * c or w1_samples * c / Fs).
% Given Fs is explicit, I will use: **d_agg = w_samples * c / Fs**.
% And verify consistency.
%
% Ref: Eq (6): x^2 + y^2 = (d1/2)^2. This assumes Monostatic at (0,0).
% Rx1 is at (0,0). Tx is at (0,0).
% Path: (0,0)->(x,y)->(0,0). Length = 2 * sqrt(x^2+y^2).
% So d1 = 2 * sqrt(x^2+y^2).
% Sq: d1^2 = 4 * (x^2+y^2) => x^2+y^2 = d1^2 / 4. Match!
% So d1 MUST be the Total Path Length.
% d1 = Time * c = (w1 / Fs) * c.

% Equation (2) from PDF: d = w*c/(2*f), where f is sampling frequency Fs
d1 = w1vec * c / Fs;
d2 = w2vec * c / Fs;

% Geometry solution from Eqs (13) and (15)
% A = 1/4 * d1^2
% B = (d2 - 1/2 * d1)^2
% x = (A - B) / (2*Delta) + Delta/2
% y = sqrt(A - x^2)

A = 0.25 * d1.^2;
B = (d2 - 0.5 * d1).^2;

xvec = (A - B) ./ (2 * Delta) + Delta / 2;

% Ensure argument for sqrt is non-negative (numerical noise might cause slightly negative)
sq_arg = A - xvec.^2;
sq_arg(sq_arg < 0) = 0; % Clamp to 0 if slightly negative
yvec = sqrt(sq_arg);

% Plotting is requested inside the function?
% "Plot the estimated path of the airplane using Matlab plot command"
% Yes.
figure;
plot(xvec, yvec, 'b.-');
xlabel('Position x [m]');
ylabel('Position y [m]');
title('Estimated Airplane Path');
grid on;
axis equal; % Important for spatial correctness
end
