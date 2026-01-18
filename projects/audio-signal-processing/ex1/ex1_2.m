% Audio Process Analysis
% Exercise 1.2
% This script loads, visualizes, plays, saves, and filters audio recordings.

clear; clc; close all;

%% 1. Load Audio Files
% Reading WAV files containing 1 minute of audio at 48KHz.
files = {'airplane.wav', 'cafe.wav', 'city.wav', 'vacuumcleaner.wav'};
titles = {'Airplane Cabin Noise', 'Cafe Background Noise', 'City Noise', 'Vacuum Cleaner Noise'};
signals = cell(1, 4);
Fs_all = zeros(1, 4);

for i = 1:length(files)
    if exist(files{i}, 'file')
        [y, Fs] = audioread(files{i});
        % Force Mono
        if size(y, 2) > 1
            y = mean(y, 2);
        end
        signals{i} = y;
        Fs_all(i) = Fs;
        fprintf('Loaded %s: Fs = %d Hz, Duration = %.2f s\n', ...
            files{i}, Fs, length(y)/Fs);
    else
        warning('File %s not found in current directory.', files{i});
        signals{i} = [];
    end
end

%% 2. Visualization
figure('Name', 'Audio Noise Signals', 'NumberTitle', 'off');

for i = 1:length(files)
    if ~isempty(signals{i})
        y = signals{i};
        Fs = Fs_all(i);
        t = (0:length(y)-1) / Fs;

        subplot(4, 1, i);
        plot(t, y);
        title(titles{i});
        xlabel('Time [s]');
        ylabel('Amplitude');
        grid on; axis tight;
    end
end

%% 3. Playback & Processing Setup
k = 1; % Select index: 1=Airplane, 2=Cafe, 3=City, 4=Vacuum
if ~isempty(signals{k})
    y_in = signals{k};
    Fs_in = Fs_all(k);

    %% 4. Filtering Implementation
    % Difference Equation:
    % a(1)y(n) = b(1)x(n) + ... - a(2)y(n-1) - ...
    %
    % We implement a Simple Low-Pass Filter (Moving Average) as a demo.
    % Equation: y[n] = 1/5 * (x[n] + x[n-1] + ... + x[n-4])
    % This means b = [0.2 0.2 0.2 0.2 0.2], a = 1.

    M = 5; % Filter order
    b = ones(1, M) / M;
    a = 1;

    % Apply Filter
    y_filtered = filter(b, a, y_in);

    % --- Visualization of Filter Effect ---
    figure('Name', 'Original vs Filtered', 'NumberTitle', 'off');
    t_segment = (0:999) / Fs_in; % View first 1000 samples for clarity

    subplot(2,1,1);
    plot(t_segment, y_in(1:1000));
    title(['Original: ' titles{k}]);
    grid on; axis tight;

    subplot(2,1,2);
    plot(t_segment, y_filtered(1:1000));
    title('Filtered (Moving Average)');
    grid on; axis tight;

    %% 5. Playback and Save
    % Normalization is crucial for playback of arbitrary signals/errors

    fprintf('\n--- Playback ---\n');
    fprintf('Playing Original...\n');
    soundsc(y_in, Fs_in);
    pause(length(y_in)/Fs_in + 1); % Wait for it to finish

    fprintf('Playing Filtered...\n');
    soundsc(y_filtered, Fs_in);

    % Save
    audiowrite(['filtered_' files{k}], y_filtered, Fs_in);
    fprintf('\nSaved filtered output to "filtered_%s"\n', files{k});
end
