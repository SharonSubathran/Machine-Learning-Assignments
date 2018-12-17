%PCA
%author: sharon subathran
%takes in data and labels and computes the principal components for the
%matrix and returns the matrix concatenated with the labels

function [pc] = pca(x,y);
m = 0;
xm = 0;
c = 0;
y = x(:,5);
m = mean(x);
m = m(:,1:4);
u = repmat(m,150,1);
u = u(:,1:4);

xm = x(:,1:4) - u;

c = cov(xm);
[eigvec_org, eigval_org]= eig(c);

eigvec = fliplr(eigvec_org);  % largest evector on 1st col 
eigval = flipud(diag(eigval_org));  % largest evalue on top 
fprintf("\neigen values =\n");
disp(eigval);
fprintf ("-----After applying PCA-----\n \n");
PC = xm * eigvec;        
PC = PC(:,1);
pc = horzcat(PC,x(:,2:5));

% fprintf ("-----After applying PCA-----\n \n");