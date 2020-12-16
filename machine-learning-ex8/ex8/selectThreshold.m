function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
epsilon = min(pval):stepsize:max(pval);
predictions = (pval < epsilon);
tp = sum((yval == 1) & (predictions == 1));
fp = sum((yval == 0) & (predictions == 1));
fn = sum((yval == 1) & (predictions == 0));
prec = tp./(tp+fp);
rec = tp./(tp+fn);
F1 = (2*prec.*rec)./(prec+rec);
[maxF1, idx] = max(F1);
bestF1 = F1(idx);
bestEpsilon = epsilon(idx);
         

end
