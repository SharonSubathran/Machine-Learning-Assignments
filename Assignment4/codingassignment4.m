%Reading corrupted_2class_iris_data

clc;
clear;
data_unshuff = dlmread('corrupted_2class_iris_dataset.dat');
index = randperm(100);
data = data_unshuff(index,:);
 
X = data(:,1:4);
r = length(X);
X = [ones(r,1) X];
y = data(:,5);
K = 10;
d = 5;
N = 100;
j=1;
J = [];
accuracy = [];
acc_avg = 0;

for k = 1:K

Xtest = X(j:j+9,:);
ytest = y(j:j+9,:);
Xtrain = X;
Xtrain(j:j+9,:) = [];
ytrain = y;
ytrain(j:j+9,:) = [];


%Gradient Descent algorithm 

w = zeros(5,1);
nu = 0.04;
lambda = 0.01;
m = length(Xtrain);
I = ones(m,1);

for (i =1:1500) 

  sigmoid = (1+exp(-(Xtrain*w))).^(-1);
  h = (sigmoid - ytrain)';
  
  w(1) = w(1) - nu*(1/m)*h*Xtrain(:,1);
  w(2) = w(2) - nu*(1/m)*h*Xtrain(:,2);
  w(3) = w(3) - nu*(1/m)*h*Xtrain(:,3);
  w(4) = w(4) - nu*(1/m)*h*Xtrain(:,4);
  w(5) = w(5) - nu*(1/m)*h*Xtrain(:,5);
 
  J(i) = 1/m*h*h';
end
count = 0;


%Testing
for (i = 1:length(Xtest))
  sigm = (1+exp(-(Xtest(i,:)*w)))^-1;
  if sigm < 0.5
     sigm = 0;
  
  else
      sigm = 1;
  
  end
  
  if sigm == ytest(i)
    count = count+1;
  end
  
  
end


end

fprintf('accuracy per iteration= %5.4f\n',count)
accuracy = count/10;
 acc_avg = mean(accuracy);
fprintf('average accuracy= %5.4f\n',acc_avg)

plot(J);
xlabel('Training Iteratons');
ylabel('Cost Funcodingction J')




   