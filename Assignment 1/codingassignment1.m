clear all; close all;
X=dlmread('housingpricedata.dat');
F=sortrows([X(:,5) X(:,2)]);
figure; scatter(F(:,1), F(:,2));
title('Scatter plot of housing prices');
xlabel('Size in square Feet');
ylabel('Price');
F=F(25:600,:);
X=F(:,1);
normalize = max(X)-min(X);
X = X/normalize;
X = [repmat(ones,length(F),1) X]; % = [1 X]
% size X is 576x2
y=F(:,2);
w = rand(2,1); % size w is 2x1

% Parameters
% X = 576x2 [1 size_of_house]
% y = 576x1
% w = 2x1

nu = 0.5; 
mse = [];
n = 576;
f1 = 0;
f2 = 0;
t = 0;
for itr=1:2000
    for i = 1:576
        f1 = f1+w(1,1)+(w(2,1)*X(i,1))-y(i,1);
        f2 = f2+w(1,1)+(w(2,1)*X(i,1))-y(i,1);
    end
    w=w(1,1)
    w(1,1)=w -(nu*(1/576)*f1);
    w(2,1)=w -(nu*(1/576)*f2);
    
    for i = 1:576
        t = t + w(1,1)+(w(2,1)*X(i,1))-y(i);
    end
    
    mse(itr,1)=(1/(2*576))*(t)^2;
    
end

yhat = X * w;
figure;
hold off;
scatter(X(:,2)*normalize,y, 50, '.'); 
hold;
plot(X(:,2)*normalize,yhat,'r');
xlabel('Size in square Feet'); 
ylabel('Price');
figure;
plot(mse,'LineWidth',2);
title('for learning rate = 0.01');
xlabel('Iterations'); 
ylabel('Error');