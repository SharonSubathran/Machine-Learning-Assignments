clear all; close all; clc;
% X is the dataset size 100x2 
%   Column 1: sepal length 
%   Column 2: sepal width 

X = dlmread('simple_iris_dataset.dat');  % Size=100x2 
N = length(X);  % N=100 

% Initialization - take 2 random samples from data set 
ctr1 = X(randi([1,N]),:); 
ctr2 = X(randi([1,N]),:);

M1 = repmat(X(randi(N),1),1,2);
M2 = repmat(X(randi(N),2),1,2);

cov1 = cov(X); 
cov2 = cov(X); 

prior1 = 0.5; 
prior2 = 0.5; 

% Misc. initialization 
idx_c1 = zeros(50,1); 
idx_c2 = zeros(50,1); 

W1 = zeros (100,1); 
W2 = zeros (100,1); 

% W1 and W2 are vectors that eventually should contain each data point's  
% membership grade relative to Gaussian 1 and Gaussian 2 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% You can define some error measure smaller than some epsilon to stop 
% the iteration.  But for now just run for 250 iterations 

for itr = 1:250 
    %estep 
    
    g1=zeros(100,1);
    g2=zeros(100,1);

    for i=1:N
        
        g1(i) = sqrt(det(cov1))^-1 * exp(-0.5*((X(i,:)' - M1')' * inv(cov1) * (X(i,:)'-M1')));
        g2(i) = sqrt(det(cov2))^-1 * exp(-0.5*((X(i,:)' - M2')' * inv(cov2) * (X(i,:)'-M2')));
        
        W1(i) = (g1(i) * prior1)/(g1(i) * prior1 + g2(i) * prior2);
        W2(i) = (g2(i) * prior2)/(g1(i) * prior1 + g2(i)* prior2);
    end
    
    %mstep
  
    prior1 = sum(W1)/100;
    prior2 = sum(W2)/100;
    
    M1 = [sum(repmat(W1,1,2).*X)/sum(W1)];
    M2 = [sum(repmat(W2,1,2).*X)/sum(W2)];
    
    cov1 = ((repmat(W1,1,2).*(X-repmat(M1,100,1)))'*(X-repmat(M1,100,1)))/sum(W1);
    cov2 = ((repmat(W2,1,2).*(X-repmat(M2,100,1)))'*(X-repmat(M2,100,1)))/sum(W2);
end

ctr1 = M1;
ctr2 = M2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
figure; hold on;  
title('Clustering with EM algorithm'); 
xlabel('Sepal Length'); 
ylabel('Sepal Width'); 

% Hard clustering assignment – W1, W2 (100x1) 
idx_c1 = find(W1 > W2); 
idx_c2 = find(W1 <= W2); 

% idx_c1 is a vector containing the indices of the points in X  
% that belong to cluster 1 (Mx1) 
% idx_c2 is a vector containing the indices of the points in X  
% that belong to cluster 2 (N-M x 1) 

% Plot clustered data with two different colors 
plot(X(idx_c1,1),X(idx_c1,2),'r.','MarkerSize',12) 
plot(X(idx_c2,1),X(idx_c2,2),'b.','MarkerSize',10)
% Plot centroid of each cluster – ctr1, ctr2  (1x2) 
plot(ctr1(:,1),ctr1(:,2), 'kx', 'MarkerSize',12,'LineWidth',2); 
plot(ctr2(:,1),ctr2(:,2), 'ko', 'MarkerSize',12,'LineWidth',2);