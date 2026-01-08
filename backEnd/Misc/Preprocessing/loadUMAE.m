function [X, M, N, D, HSI, GT, Y, extra1, extra2] = loadUMAE(datasetName, latentPath)

if nargin < 2 || isempty(latentPath)
   error('latentDir is required. Call loadMAE(datasetName, latentDir) with a non-empty latentPath.');
end 

[~, M, N, ~, ~, GT, Y, extra1, extra2] = loadHSI(datasetName);

assert(exist(latentPath, 'file') == 2, 'Latent file not found: %s', latentPath);

S = load(latentPath);

assert(isfield(S, 'latent_spatial'), ...
    'latent_spatial missing in latent file: %s', latentPath);

Z_map = double(S.latent_spatial); 
[Hc, Wc, D] = size(Z_map);

assert(Hc == M && Wc == N, ...
    'Latent size mismatch: latent=[%d %d], dataset=[%d %d]', Hc, Wc, M, N);

HSI = Z_map;

X = reshape(Z_map, [], D);

X = zscore(X, 0, 1); 

n = vecnorm(X, 2, 2);
n(n < 1e-6) = 1e-6;
X = X ./ n;

end
