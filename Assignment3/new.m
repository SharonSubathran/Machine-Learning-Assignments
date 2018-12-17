data_unshuff = dlmread("corrupted_iris_dataset.dat");
%data_unshuff = dlmread("iris_dataset.dat");
N = 150;    %number of samples
NC = 50;    %size of each class
K = 10;
%seed = 150; rand('seed', seed);
index = randperm(N); 
data = data_unshuff(index,:);
foldsize = N/K;

c = 1;
%acc = 0;
perc = 0;
ff = [];
train_a = [];
train_b = [];
train = [];
test = [];
g1 = 0;
g2 = 0;
g3 = 0;
data_1 = [];
data_2 = [];
data_3 = [];

predicted = [];

%separating into training and testing data

for z=1:foldsize:135
    acc = 0;
    data_1 = [];
    data_2 = [];
    data_3 = [];
    
    mean_mle_1 = [];
    mean_mle_2 = [];
    mean_mle_3 = [];
    
    cov_mle1 =[];
    cov_mle2 =[];
    cov_mle3 =[];

    test = data((z:z+foldsize-1),(1:5));
    
    if(z ==135)
       train = data(1:134,1:5);
    else

        if(z ~= 1)
            train_a = data(1:z-1,(1:5));
            train_b = data(foldsize+1:N,(1:5));
            train = vertcat(train_a,train_b);

        else 
            train = data(foldsize+1:N,(1:5));
        end
        
    end
    

    %separating into class 1, 2, 3
    
    for j=1:135
        
        if train(j,5)==1
            p = train(j,5);
            data_1 = vertcat(data_1,train(j,1:4));
        end
        
        if train(j,5)==2
            data_2 = vertcat(data_2,train(j,1:4));
        end
        
        if train(j,5)==3
            data_3 = vertcat(data_3,train(j,1:4));
        end
    end
    
    %Mean MLE
    mean_mle_1 = mean(data_1(:,1:4));
    mean_mle_2 = mean(data_2(:,1:4));
    mean_mle_3 = mean(data_3(:,1:4));
    
    %Cov MLE
    cov_mle1 = covmle(data_1(:,1:4), mean_mle_1');
    cov_mle2 = covmle(data_2(:,1:4), mean_mle_2');
    cov_mle3 = covmle(data_3(:,1:4), mean_mle_3');
    
    l = size(test,1)
    
    for z=1:l
        a = (test(z,(1:4)) - (mean_mle_1))'
        b = (test(z,(1:4)) - (mean_mle_2))'
        c = (test(z,(1:4)) - (mean_mle_3))'
        g1 = -0.5 * transpose(a) * inv(cov_mle1) * (a) - (0.5 * log(det(cov_mle1))) + log(1/3);
        g2 = -0.5 * transpose(b) * inv(cov_mle2) * (b) - (0.5 * log(det(cov_mle2))) + log(1/3);
        g3 = -0.5 * transpose(c) * inv(cov_mle3) * (c) - (0.5 * log(det(cov_mle3))) + log(1/3);
        
        gm = [g1, g2, g3]
        [gmax,ff] = max(gm);
        
        predicted(z) = ff
            if (predicted(z) == test(z,5))
                acc = acc+1;
            end
   
    end
   
    prec = acc/15;
    fprintf('Accuracy = %5.4f\n', prec);
end

