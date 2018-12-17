%%% Bayes Decision Theoretic %%%
% MLE Covariance matrix 

function [cov] = covmle(tr_data, u);

X = tr_data;
u = u';
n = length(X);
M = repmat(u,n,1);

cov = ((X-M)' * (X-M)) / n;

%  M = repmat(mean_mle_1,43,1)
%  n = length(data_1(:,1:4));
%  cov = ((data_1(:,1:4)-M)' * (data_1(:,1:4)-M)) / n;