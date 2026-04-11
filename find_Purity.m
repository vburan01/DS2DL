function summary = find_Purity(resultsFolder, gtFile)
% FIND_PURITY
%   Scans a folder of S2DL result .mat files (with Cs, NNs, prctiles)
%   and a corresponding GT file (KSC_gt.mat or Botswana_gt.mat)
%   and computes:
%       - cluster Purity
%       - NMI
%   and stores the best global result in the folder of .mat files. 
%   summary = find_Purity(resultsFolder, gtFile)

    if nargin < 1 || isempty(resultsFolder)
        resultsFolder = pwd;
    end


    % Loads GT
    
    gtStruct = load(gtFile);

    if isfield(gtStruct, 'KSC_gt')
        GT = gtStruct.KSC_gt;
        gtVarName = 'KSC_gt';
    elseif isfield(gtStruct, 'Botswana_gt')
        GT = gtStruct.Botswana_gt;
        gtVarName = 'Botswana_gt';
    else
        error('GT file %s must contain KSC_gt or Botswana_gt.', gtFile);
    end

    fprintf('Using GT variable "%s" from %s\n', gtVarName, gtFile);

    gtVecFull = GT(:);  

    files = dir(fullfile(resultsFolder, '*.mat'));

    if isempty(files)
        error('No .mat files found in folder: %s', resultsFolder);
    end

    % Initialize tracking of global bests

    globalBestPurity = -Inf;
    globalBestNMI    = -Inf;

    globalBestPurityInfo = struct('purity', [], 'fileName', '', ...
                                  'NN', [], 'prctile', [], ...
                                  'iNN', [], 'jPrc', []);
    globalBestNMIInfo    = struct('NMI', [], 'fileName', '', ...
                                  'NN', [], 'prctile', [], ...
                                  'iNN', [], 'jPrc', []);

    perFile = struct('fileName', {}, ...
                     'bestPurity', {}, 'bestNMI', {}, ...
                     'iNN_purity', {}, 'jPrc_purity', {}, ...
                     'iNN_NMI', {}, 'jPrc_NMI', {}, ...
                     'NN_purity', {}, 'prctile_purity', {}, ...
                     'NN_NMI', {}, 'prctile_NMI', {});

    % Loop over files

    for f = 1:numel(files)
        fname = fullfile(resultsFolder, files(f).name);
        S = load(fname);

        %requires expected fields
        if ~isfield(S, 'Cs') || ~isfield(S, 'M') || ~isfield(S, 'N') ...
                             || ~isfield(S, 'NNs') || ~isfield(S, 'prctiles')
            fprintf('Skipping %s (missing Cs/M/N/NNs/prctiles)\n', files(f).name);
            continue;
        end

        Cs       = S.Cs;           % (M*N) x nNN x nPrc
        [numPix, nNN, nPrc] = size(Cs);
        NNs      = S.NNs;
        prctiles = S.prctiles;

        %check GT size consistency
        if numPix ~= numel(gtVecFull)
            fprintf('Warning: GT size mismatch in %s. Skipping.\n', files(f).name);
            continue;
        end

        gtVec = gtVecFull;

        %per-file bests
        bestPurity_file = -Inf;
        bestNMI_file    = -Inf;

        idxPur_i = NaN; idxPur_j = NaN;
        idxNMI_i = NaN; idxNMI_j = NaN;

        %Loop over all (NN, percentile) combinations
        for iNN = 1:nNN
            for jP = 1:nPrc
                predVec = Cs(:, iNN, jP);

                % Purity 
                p = purity_ignore_bg(predVec, gtVec);

                % NMI 
                nmiVal = nmi_ignore_bg(predVec, gtVec);

                %Track per-file bests
                if p > bestPurity_file
                    bestPurity_file = p;
                    idxPur_i = iNN;
                    idxPur_j = jP;
                end
                if nmiVal > bestNMI_file
                    bestNMI_file = nmiVal;
                    idxNMI_i = iNN;
                    idxNMI_j = jP;
                end

                %Track global bests
                if p > globalBestPurity
                    globalBestPurity = p;
                    globalBestPurityInfo.purity   = p;
                    globalBestPurityInfo.fileName = files(f).name;
                    globalBestPurityInfo.iNN      = iNN;
                    globalBestPurityInfo.jPrc     = jP;
                    globalBestPurityInfo.NN       = NNs(iNN);
                    globalBestPurityInfo.prctile  = prctiles(jP);
                end

                if nmiVal > globalBestNMI
                    globalBestNMI = nmiVal;
                    globalBestNMIInfo.NMI      = nmiVal;
                    globalBestNMIInfo.fileName = files(f).name;
                    globalBestNMIInfo.iNN      = iNN;
                    globalBestNMIInfo.jPrc     = jP;
                    globalBestNMIInfo.NN       = NNs(iNN);
                    globalBestNMIInfo.prctile  = prctiles(jP);
                end
            end
        end

        % Store per-file best result
        perFile(end+1).fileName       = files(f).name; %#ok<AGROW>
        perFile(end).bestPurity       = bestPurity_file;
        perFile(end).bestNMI          = bestNMI_file;

        perFile(end).iNN_purity       = idxPur_i;
        perFile(end).jPrc_purity      = idxPur_j;
        perFile(end).iNN_NMI          = idxNMI_i;
        perFile(end).jPrc_NMI         = idxNMI_j;

        perFile(end).NN_purity        = NNs(idxPur_i);
        perFile(end).prctile_purity   = prctiles(idxPur_j);
        perFile(end).NN_NMI           = NNs(idxNMI_i);
        perFile(end).prctile_NMI      = prctiles(idxNMI_j);

        fprintf('File %s:\n', files(f).name);
        fprintf('  best purity   = %.4f at NN=%d, prctile=%d\n', ...
            bestPurity_file, perFile(end).NN_purity, perFile(end).prctile_purity);
        fprintf('  best NMI      = %.4f at NN=%d, prctile=%d\n\n', ...
            bestNMI_file, perFile(end).NN_NMI, perFile(end).prctile_NMI);
    end


    % print final summary
    
    if globalBestPurity < 0
        error('No valid scores computed. Check folder and GT.');
    end

    fprintf('\n Global Best Results \n');
    fprintf('Best purity      = %.4f (file=%s, NN=%d, prctile=%d)\n', ...
        globalBestPurity, globalBestPurityInfo.fileName, ...
        globalBestPurityInfo.NN, globalBestPurityInfo.prctile);
    fprintf('Best NMI         = %.4f (file=%s, NN=%d, prctile=%d)\n', ...
        globalBestNMI, globalBestNMIInfo.fileName, ...
        globalBestNMIInfo.NN, globalBestNMIInfo.prctile);


    summary.globalBestPurity = globalBestPurityInfo;
    summary.globalBestNMI    = globalBestNMIInfo;
    summary.perFile          = perFile;
end
