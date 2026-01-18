function w = linpred_wiener(x, r, L)
    % LINPRED_WIENER Solves the linear estimation problem of order L.
    %
    %   w = LINPRED_WIENER(x, r, L) computes the optimal Wiener coefficients w
    %   that minimize the mean squared error E[(x_n - w^T * r_vec_n)^2],
    %   where r_vec_n is the vector of L past reference samples.
    %
    %   INPUTS:
    %       x : Target sequence X_n (column vector, length N)
    %       r : Measurement/reference sequence R_n (column vector, length N)
    %       L : Predictor order (scalar)
    %
    %   OUTPUT:
    %       w : Optimal Wiener predictor coefficients (Lx1 column vector)
    %
    %   The function estimates the correlation matrix R and cross-correlation
    %   vector p empirically from the available data, then solves R*w = p.
    
    % Ensure inputs are column vectors
    x = x(:);
    r = r(:);
    N = length(x);
    
    if length(r) ~= N
        error('Input vectors x and r must have the same length.');
    end
    
    if N <= L
        error('Data length N must be greater than predictor order L.');
    end
    
    % --- Implementation using Lag Matrix (Toeplitz-like structure) ---
    % We want to predict x(n) using r(n-1), r(n-2), ..., r(n-L).
    % Alternatively, if we align it such that we define r_vec_n = [r(n); r(n-1); ...],
    % we need to be careful with the indices asked by the problem.
    %
    % Problem Statement 1.4(e) defines Y_n = [R_{n-1} ... R_{n-L}]^T.
    % This implies we are using strictly past samples to predict X_n.
    %
    % We construct the regressor matrix A (size: (N-L) x L)
    % Row k corresponds to time index n = L+k.
    % A(k, :) = [r(L+k-1), r(L+k-2), ..., r(k)]
    
    % Number of valid samples we can predict (starting from n = L+1)
    n_valid = N - L;
    
    try
        % Build Toeplitz matrix using 'toeplitz' for efficiency or manual construction
        % Column 1: r(L : N-1)
        % Row 1:    [r(L), r(L-1), ..., r(1)]
        col1 = r(L : N-1);
        row1 = r(L : -1 : 1);
        
        % The regression matrix A.
        % A * w \approx x(L+1 : N)
        A = toeplitz(col1, row1); 
        
        % Target vector aligned with the rows of A
        b = x(L+1 : N);
        
        % Solve normal equations: (A'A)w = A'b
        % R_hat = (1/n_valid) * (A'A)
        % p_hat = (1/n_valid) * (A'b)
        % w = R_hat \ p_hat;
        
        % Using the backslash operator is more numerically stable than explicit inversion
        w = (A' * A) \ (A' * b);
        
    catch ME
        % Fallback loop if memory is an issue (though N=480k is fine for toeplitz)
        warning('Vectorized construction failed, falling back to simple method. Error: %s', ME.message);
        A = zeros(n_valid, L);
        for k = 1:n_valid
             idx = L + k; % Current time index n
             % Regressors: r(n-1) ... r(n-L)
             A(k, :) = r(idx-1 : -1 : idx-L)'; 
        end
        b = x(L+1 : N);
        w = (A' * A) \ (A' * b);
    end
end
