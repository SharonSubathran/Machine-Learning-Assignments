clear all; close all;
X=dlmread('housing_price_data.dat');
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



for i = 1:576
    w_ols=(pinv(X'*X))*(X'*y);
    yhat=(w_ols(1)*X(:,1)+(w_ols(2)*X(:,2)));
end



figure; hold off; scatter(X(:,2)*normalize,y, 50, '.'); hold; plot(X(:,2)*normalize,yhat,'r'); xlabel('Size in square Feet'); ylabel('Price'); 