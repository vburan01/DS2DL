function p = purity_ignore_bg(pred, gt)
    pred = pred(:);
    gt   = gt(:);

    % Ignore background (GT == 0)
    valid = gt > 0;
    pred  = pred(valid);
    gt    = gt(valid);

    if isempty(pred)
        p = NaN;
        return;
    end

    tab = crosstab(pred, gt);  % rows: clusters, cols: classes
    p   = sum(max(tab, [], 2)) / sum(tab(:));
end
