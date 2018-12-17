clear all; 
close all; 
%   X is the dataset size 100x2 
%   Column 1: sepal length 
%   Column 2: sepal width 

X = dlmread('simple_iris_dataset.dat'); 
N = length(X);  % N=100 

% Initialization - take 2 random samples from data set 
ctr1 = X(randi([1,N]),:); 
ctr2 = X(randi([1,N]),:); 

% Misc. initialization 
idx_c1 = zeros(50,1); 
idx_c2 = zeros(50,1);
cl1 = zeros(100,2); 
cl2 = zeros(100,2); 
itr = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% K-means algorithm 

for itr = 1:100
    idx_c1 = [];
    idx_c2 = [];
    
    for i = 1:100
        
        d1 = (ctr1(1,1)-X(i,1))^2 + (ctr1(1,2)-X(i,2))^2;
        d2 = (ctr2(1,1)-X(i,1))^2 + (ctr2(1,2)-X(i,2))^2;
        
        if(d1 < d2)
            idx_c1 = [idx_c1 ; i];
            
        else
            idx_c2 = [idx_c2 ; i];
            
        end 
        
    end
    
    temp1 = ctr1;
    temp2 = ctr2;
    
    ctr1 = [mean(X(idx_c1' , 1)), mean(X(idx_c1', 2))];
    ctr2 = [mean(X(idx_c2' , 1)), mean(X(idx_c2', 2))];
    
    if isequal(temp1, ctr1) && isequal(temp2, ctr2)
        fprintf("Converges after %d iterations\n",itr);
        break
    end
    
end

% Print the number of iterations required for the algorithm to converge 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure; 
hold on; 
xlabel('Sepal Length'); 
ylabel('Sepal Width'); 

% idx_c1 is a vector containing the indices of the points in X 
% that belong to cluster 1 (Mx1) 
% idx_c2 is a vector containing the indices of the points in X 
% that belong to cluster 2 (N-M x 1), N=100 

% Plot clustered data with two different colors; X is 100x2 
plot(X(idx_c1,1),X(idx_c1,2),'r.','MarkerSize',12) 
plot(X(idx_c2,1),X(idx_c2,2),'b.','MarkerSize',10) 

% Plot centroid of each cluster 
plot(ctr1(:,1),ctr1(:,2), 'kx', 'MarkerSize',12,'LineWidth',2); 
plot(ctr2(:,1),ctr2(:,2), 'ko', 'MarkerSize',12,'LineWidth',2); 










