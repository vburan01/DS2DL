%specify dataset, no. training pixels and Masking Ratio

clc; clearvars 
profile off; profile on;

dsChoice = input("Enter dataset (1 = Botswana, 2 = KSC): ");
if ~isscalar(dsChoice) || ~ismember(dsChoice, [1 2])
    error('Invalid choice. Enter 1 for Botswana or 2 for KSC.');
end

if dsChoice == 1
    datasetName = 'Botswana';
else
    datasetName = 'KSC';
end

numTrainPixels = input("Enter number of training pixels: ");
if ~isscalar(numTrainPixels) || ~isfinite(numTrainPixels) || numTrainPixels <= 0
    error('Number of training pixels must be a positive scalar.');
end

maskStr = strtrim(input("Enter masking ratio: ", 's'));
maskRatio = str2double(maskStr);
if ~isscalar(maskRatio) || ~isfinite(maskRatio) || maskRatio <= 0 || maskRatio >= 1
    error('Masking ratio must be a number strictly between 0 and 1.');
end

thisDir = fileparts(mfilename('fullpath'));
outputsDir = fullfile(thisDir,'outputs');

latentName = sprintf('%s_FPS%d_MR%s.mat', datasetName, numTrainPixels, maskStr);
latentPath  = fullfile(outputsDir, latentName);   

if ~isfile(latentPath)
    error('Latent file not found: %s', latentPath);
end

[~, M, N, ~, HSI_raw, GT, Y, ~, ~] = loadHSI(datasetName);

[X, ~, ~, D, ~, ~, ~, ~, ~] = loadUMAE(datasetName, latentPath);

HSI = HSI_raw;

tRun = tic;

[Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 51);
Idx_NN = Idx_NN(:, 2:51);
Dist_NN = Dist_NN(:, 2:51);

Hyperparameters.SpatialParams.ImageSize = [M, N];
Hyperparameters.NEigs          = 10;
Hyperparameters.NumDtNeighbors = 200;
Hyperparameters.Beta           = 2;
Hyperparameters.Tau            = 1e-5;
Hyperparameters.K_Known        = length(unique(Y));
Hyperparameters.Tolerance      = 1e-8;

clc; profile off;
disp('Dataset preloaded.');

globalBest.score  = -inf;
globalBest.labels = [];
globalBest.meta = struct('Dataset', datasetName, ...
                         'NumSP', NaN, 'nk', NaN, 'R', NaN);

for nk = [3, 5]
    Hyperparameters.LocalBackbones = 1;
    Hyperparameters.nk = nk;
    disp('value of nk is:');
    disp(nk);

    for p = 1000:200:2400
  
        map   = seg_ERS(HSI, 0, p);
        spSeg = double(map);

        if ~isempty(spSeg) && min(spSeg(:)) == 1
            spSeg = spSeg - 1;
        end
        numSuperpixels = max(spSeg(:)) + 1;

        Hyperparameters.Superpixel.map = spSeg;
        Hyperparameters.Superpixel.num = numSuperpixels;

        currentPerf = 0; maxSum = NaN;

        for l = 2:2:30
            NNs       = 30:10:50;
            prctiles  = [30,50,70,90];
            numReplicates = 1; 

            OAs     = NaN*zeros(length(NNs), length(prctiles), numReplicates);
            kappas  = NaN*zeros(length(NNs), length(prctiles), numReplicates);
            AAs     = NaN*zeros(length(NNs), length(prctiles), numReplicates);
            Cs      = zeros(M*N, length(NNs), length(prctiles), numReplicates);

            disp('Set hyperparameter grid to begin grid search');
            Hyperparameters.SpatialParams.SpatialRadius = l;

            for i = 1:length(NNs)
                for j = 1:length(prctiles)
                    for k = 1:numReplicates

                        Hyperparameters.DiffusionNN = NNs(i);
                        Hyperparameters.DensityNN   = NNs(i);
                        Hyperparameters.Sigma0 = prctile( ...
                            Dist_NN(Dist_NN(:,1:NNs(i))>0), prctiles(j), 'all');

                        density     = KDE_large(Dist_NN, Hyperparameters);
                        Clusterings = S2DL(X, density, Hyperparameters);

                        if isfield(Clusterings, 'Labels')
                            [OAs(i,j,k), kappas(i,j,k), tIdx, ~, ~, AAs(i,j,k)] = ...
                                calcPerformance(Y, Clusterings, ~strcmp('JasperRidge', datasetName));
                            C = Clusterings.Labels(:, tIdx);
                            Cs(:, i, j, k) = C;
                        end

                        disp('S2DL: ');
                        disp([i/length(NNs), j/length(prctiles), k/numReplicates, currentPerf]);
                    end

                    currentPerf = [max(nanmean(OAs,3),[],'all'), ...
                                   max(nanmean(kappas,3),[],'all'), ...
                                   max(nanmean(AAs,3),[],'all')];
                end

                [n1,n2] = size(nanmean(OAs,3));
                [maxSum, kbest] = max(reshape(nanmean(OAs+kappas+AAs,3), n1*n2, 1));
                [iBest, jBest] = ind2sub(size(mean(OAs+kappas+AAs,3)), kbest);

                save(sprintf('DS2DL_%s_%dSP_nk%d_R%d.mat', ...
                             datasetName, numSuperpixels, ...
                             Hyperparameters.nk, Hyperparameters.SpatialParams.SpatialRadius), ...
                     'M','N','OAs','kappas','AAs','Cs','NNs','prctiles','numReplicates','maxSum');

                blockScore = maxSum;
                if blockScore > globalBest.score
                    bestLabels = Cs(:, iBest, jBest, 1);
                    globalBest.labels = reshape(bestLabels, [M N]);
                    globalBest.score  = blockScore;
                    globalBest.meta.NumSP = numSuperpixels;
                    globalBest.meta.nk    = Hyperparameters.nk;
                    globalBest.meta.R     = Hyperparameters.SpatialParams.SpatialRadius;
                end
            end
        end
    end
end

fprintf('best accuracy run: DS2DL_%s_%dSP_nk%d_R%d.mat' , ...
    globalBest.meta.Dataset, globalBest.meta.NumSP, globalBest.meta.nk, globalBest.meta.R, globalBest.score);

