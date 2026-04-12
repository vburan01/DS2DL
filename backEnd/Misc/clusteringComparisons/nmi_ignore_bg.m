function nmiVal = nmi_ignore_bg(pred, gt)
    pred = pred(:);
    gt   = gt(:);

    valid = gt > 0;
    pred  = pred(valid);
    gt    = gt(valid);

    if isempty(pred)
        nmiVal = NaN;
        return;
    end

    [nmiVal, ~, ~, ~] = nmi_labels(pred, gt);
end

function [NMI, MI, Hu, Hv] = nmi_labels(U, V)
% NMI between label vectors U and V (1D, same length).
    assert(numel(U) == numel(V));
    n = numel(U);

    U = reshape(U,1,n);
    V = reshape(V,1,n);

    % Make labels start at 1
    l = min(min(U),min(V));
    U = U - l + 1;
    V = V - l + 1;
    k_max = max(max(U),max(V));

    idx = 1:n;
    Mu = sparse(idx, U, 1, n, k_max, n);
    Mv = sparse(idx, V, 1, n, k_max, n);

    Puv = nonzeros(Mu' * Mv / n);      % joint distribution
    Huv = -dot(Puv, log2(Puv));        % joint entropy

    Pu = nonzeros(mean(Mu,1));         % marginal U
    Pv = nonzeros(mean(Mv,1));         % marginal V

    Hu = -dot(Pu, log2(Pu));
    Hv = -dot(Pv, log2(Pv));

    MI = Hu + Hv - Huv;

    if Hu > 0 && Hv > 0
        NMI = 2 * MI / (Hu + Hv);      % symmetric NMI
        NMI = max(0, NMI);             % clamp small negatives
    else
        NMI = 0;
    end
end
