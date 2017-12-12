% MATLAB program for exercise 3 in course 02457
% This program is for part 3 out of 3 
%
% "main3c" illustrates the use of the Fisher discriminant
% to reduce the dimensionality of a classification problem.
% 
% The parameters that should be changed are
%   mu1  : The true mean of the class C1
%   mu2  : The true mean of the class C2
%   SIGMA: The covariance matrix of each of the classes
%   p1   : P(C1)  probabilty of class C1
%   N    : number of points in the training-set

%   (c) Karam Sidaros, September 1999.
%  Uses randmvn.m randbin.m arrow.m
%

%%%%%%%%%%%%%%%%%%%%%%%%% Part 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Fisher Discriminant %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off
%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  mu1=[5 4]';	    % true mean value
  mu2=[4 -1]';	    % true mean value

  SIGMA=[2   -1.7 
    -1.7 3  ];   % true covariance matrix

  p1 = 0.3;          % P(C1)  probabilty of class 1
  p2 = 1-p1;

  N=4000;            % number of points in density

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d1 = randmvn(mu1,SIGMA,N);  % A random sample
d2 = randmvn(mu2,SIGMA,N);  % A random sample

c1 = randbin(p1,1,N);        % Occurrence of class 1
c2 = 1 - c1;                 % Occurrence of class 2

X  = d1.*[c1;c1] + d2.*[c2; c2];

T = c1 - c2;                 % y=1 for C1 and -1 for C2

N1 = sum(c1);                % no. of occurence of C1
N2 = sum(c2);

m1 = sum(X.*[c1;c1],2)/N1;
m2 = sum(X.*[c2;c2],2)/N2;

SW1 = ((X-repmat(m1,1,N)).*[c1;c1])*((X-repmat(m1,1,N))'.*[c1;c1]');
SW2 = ((X-repmat(m2,1,N)).*[c2;c2])*((X-repmat(m2,1,N))'.*[c2;c2]');

SW = SW1 + SW2;

w = inv(SW)*(m2-m1);
w = w/norm(w);


Y = w'*X;

%%%%%%%%%%%%%% Eigenvector Transformation %%%%%%%%%%

mu_    = mean(X,2);       % Estimated mean
SIGMA_ = cov(X');         % Estimated covariance matrix
[U lambda] = eig(SIGMA_); % Eigenvectors and eigenvalues
 
X_ = U'*(X-repmat(mu_,1,N));  % Coordinate transformation

fprintf('Calculating histograms ... ');
  nbins = 50;                   % # bins in histogram 
  [n1_ x1_] = hist(X_(1,:),nbins); % Marginal Histograms
  [n2_ x2_] = hist(X_(2,:),nbins);
  n1_ = n1_/sum(n1_);  
  n2_ = n2_/sum(n2_);  
  [n_y y1_] = hist(Y,nbins);  % histogram of Fisher transformed 
  n_y = n_y/sum(n_y);  
fprintf('Done\n\r');

%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%
disp('The calculated weight-vector is ');
w
disp('The eigenvectors and eigenvalues are ');
U
lambda
N
%%%%%%%%%%%%%%%%%%% Plotting 3D %%%%%%%%%%%%%%%%%%%

figure(1)
clf
subplot 221
  plot(X(1,logical(c1)),X(2,logical(c1)),'b.');     %Class 1
  hold on
  plot(X(1,logical(c2)),X(2,logical(c2)),'r.');     %Class 2
  ax=axis;
  axis equal;
  axis(ax);
  h = get(gca,'position');
  for j=1:2
    harrow(j) =  arrow(mu_(1)+[0 lambda(j,j)*U(1,j)],mu_(2)+[0 lambda(j,j)*U(2,j)],0.04*diff(xlim),2,0.25,'w');
  end
  w1=w*0.3*diff(xlim);  
  harrow(3) =  arrow(mu_(1)+[0 w1(1)],mu_(2)+[0 w1(2)],0.04*diff(xlim),2,0.25,'w');
  set(harrow(3),'facecolor',0.85*[1 1 1]);
  xlabel('x_1')
  ylabel('x_2')  
  title('Scatter plot of data-set');
  lh = legend(harrow(2:3),'Eigenvectors','Fisher');  
  temp = get(lh,'position');
  temp(1) = 0.01;
  set(lh,'position',temp); 
  legend('C_1','C_2')
subplot 222
  bar(y1_, n_y,1,'w');
  xlabel('y');
  ylabel('p(y)');
  title('Marginal Distribution after Fisher transf.');  
subplot 223
  bar(x1_,n1_,1,'w');
  ylim([0 max(n1_)*1.02]);    
  xlabel('^~x_1')
  ylabel('p(^~x_1)')
  title('Marginal Distribution after Eigenv. transf.');    
subplot 224
  bar(x2_,n2_,1,'w');
  ylim([0 max(n2_)*1.02]);    
  xlabel('^~x_2')  
  ylabel('p(^~x_2)')
  title('Marginal Distribution after Eigenv. transf.');    


