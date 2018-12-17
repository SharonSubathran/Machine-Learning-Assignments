
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% STEP 1 %%%%%%%%%%%%%%%%%%%%%
%data_unshuff = dlmread("corrupted_iris_dataset.dat");
data_unshuff = dlmread("iris_dataset.dat");
N = 150;    %number of samples
NC = 50;    %size of each class
K = 10;
%seed = 150; rand('seed', seed);
index = randperm(N); 
data = data_unshuff(index,:);
foldsize = N/K;
pc = 0;

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
p = 0;
p2 = 0;
predicted = [];
s = [];

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
            train_b = data(z+foldsize:N,(1:5));
            train = vertcat(train_a,train_b);

        else 
            train = data(foldsize+1:N,(1:5));
        end
        
    end
    
    %separating into class 1, 2, 3
    
    for j=1:135
        
        if train(j,5)==1
            %p = train(j,5);
            data_1 = vertcat(data_1,train(j,1:4));
        end
        
        if train(j,5)==2
            data_2 = vertcat(data_2,train(j,1:4));
        end
        
        if train(j,5)==3
            data_3 = vertcat(data_3,train(j,1:4));
        end
    end
    
    %using only feature 1
    testx = test(:,1);
    data1x = data_1 (:,1);
    data2x = data_2 (:,1);
    data3x = data_3 (:,1);
       
    %Mean MLE
    mean_mle_1 = mean(data_1(:,1));
    mean_mle_2 = mean(data_2(:,1));
    mean_mle_3 = mean(data_3(:,1));
    
    %Cov MLE
    cov_mle1 = covmle(data_1(:,1), mean_mle_1');
    cov_mle2 = covmle(data_2(:,1), mean_mle_2');
    cov_mle3 = covmle(data_3(:,1), mean_mle_3');
    
    l = size(test,1);
    
    for i=1:l
        a = (test(i,(1)) - (mean_mle_1))';
        b = (test(i,(1)) - (mean_mle_2))';
        c = (test(i,(1)) - (mean_mle_3))';
        g1 = -0.5 * transpose(a) * inv(cov_mle1) * (a) - (0.5 * log(det(cov_mle1))) + log(1/3);
        g2 = -0.5 * transpose(b) * inv(cov_mle2) * (b) - (0.5 * log(det(cov_mle2))) + log(1/3);
        g3 = -0.5 * transpose(c) * inv(cov_mle3) * (c) - (0.5 * log(det(cov_mle3))) + log(1/3);
        
        gm = [g1, g2, g3];
        [gmax,ff] = max(gm);
        
        predicted(i) = ff;
        test(:,5)';
            if (predicted(i) == test(i,5))
                acc = acc+1;
            end
    end
    prec(i) = acc/15;         
    fprintf('Classification accuracy= %5.4f\n',prec(i));
    p = [p;prec(i)];
end

s = sum(p)/10;
fprintf('\nAverage accuracy = %5.4f\n', s); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% STEP 2 %%%%%%%%%%%%%%%%%%%%%
%Dimensionality Reduction

d = data(:,5);
pc = pca(data, d);
s2 = 0;
z = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% STEP 3 %%%%%%%%%%%%%%%%%%%%%
%Substituting thr first feature with pc1
for z=1:foldsize:135

    acc2 = 0;
    data_1 = [];
    data_2 = [];
    data_3 = [];
    
    mean_mle_1 = [];
    mean_mle_2 = [];
    mean_mle_3 = [];
    
    cov_mle1 =[];
    cov_mle2 =[];
    cov_mle3 =[];

    test = pc((z:z+foldsize-1),(1:5));
    
    if(z ==135)
       train = pc(1:134,1:5);
    else

        if(z ~= 1)
            train_a = pc(1:z-1,(1:5));
            train_b = pc(z+foldsize:N,(1:5));
            train = vertcat(train_a,train_b);

        else 
            train = pc(foldsize+1:N,(1:5));
        end
        
    end
    
    %separating into class 1, 2, 3
    
    for j=1:135
        
        if train(j,5)==1
            %p = train(j,5);
            data_1 = vertcat(data_1,train(j,1:4));
        end
        
        if train(j,5)==2
            data_2 = vertcat(data_2,train(j,1:4));
        end
        
        if train(j,5)==3
            data_3 = vertcat(data_3,train(j,1:4));
        end
    end
    
    %using only feature 1
    testx = test(:,1);
    data1x = data_1 (:,1);
    data2x = data_2 (:,1);
    data3x = data_3 (:,1);
       
    %Mean MLE
    mean_mle_1 = mean(data_1(:,1));
    mean_mle_2 = mean(data_2(:,1));
    mean_mle_3 = mean(data_3(:,1));
    
    %Cov MLE
    cov_mle1 = covmle(data_1(:,1), mean_mle_1');
    cov_mle2 = covmle(data_2(:,1), mean_mle_2');
    cov_mle3 = covmle(data_3(:,1), mean_mle_3');
    
    l = size(test,1);
    
    for i=1:l
        a = (test(i,(1)) - (mean_mle_1))';
        b = (test(i,(1)) - (mean_mle_2))';
        c = (test(i,(1)) - (mean_mle_3))';
        g1 = -0.5 * transpose(a) * inv(cov_mle1) * (a) - (0.5 * log(det(cov_mle1))) + log(1/3);
        g2 = -0.5 * transpose(b) * inv(cov_mle2) * (b) - (0.5 * log(det(cov_mle2))) + log(1/3);
        g3 = -0.5 * transpose(c) * inv(cov_mle3) * (c) - (0.5 * log(det(cov_mle3))) + log(1/3);
        
        gm = [g1, g2, g3];
        [gmax,ff] = max(gm);
        
        predicted2(i) = ff;
        test(:,5)';
            if (predicted2(i) == test(i,5))
                acc2 = acc2+1;
            end
    end
    prec2(i) = acc2/15;         
    fprintf('Classification accuracy= %5.4f\n',prec2(i));
    p2 = [p2;prec2(i)];
    %s2 = s2 + p2;
end

s2 = sum(p2)/10;
fprintf('\nAverage accuracy = %5.4f\n', s2); 
