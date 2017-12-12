% Nonlinear signalprocessing COURSE 02457
%
%  MAIN for exercise 1   (c) Lars Kai Hansen, 
%
%  This exercise has three parts:
% 
%  1) Explore uni-and multivariate normal distributions
%  2) Illustrate Bayes decision theory and 
%  3) Confusion in decision processes
%
%  Uses norm1d.m, probconfus.m
%

disp('We use norm1d.m to compute the density (p1)')
disp('The density is evaluated in a set of points and') 
disp('and as histogram (p2)')
disp('evaluated with a given width')

mu=0.0;       % true mean value
sigma2=1.0;   % true variance
xmin=-5.0;    % min x value
xmax=5.0;     % max x valued
Npdf=100;        % number of points in density
dx=0.4;       % bin width for histogram
[x1,p1,x2,p2]=norm1d(mu,sigma2,xmin,xmax,Npdf,dx);
figure(1), subplot(2,1,1), plot(x1,p1,'b'), title('density of the 1D normal dist')
figure(1), subplot(2,1,2), bar(x2,p2,'b'), title('histogram of the 1D normal dist')

% draw a set of normal variates using MATLAB randn function
N=100; % number of samples
y1=sqrt(sigma2)*randn(N,1)+mu*ones(N,1);
y2=hist(y1,x2);
y2=y2/sum(y2);
figure(2), subplot(2,1,1),  bar(x2,y2,'r'),hold on, plot(x2,p2,':'),hold off, title('histogram of samples and "true" histogram')
figure(2), subplot(2,1,2),  bar(x2,p2,'b'), title('"true" histogram')
disp('print figure 1 and 2 for the report')
disp('-explain the difference between the density and the histogram')
disp('compute and comment on the sums of the density values')
disp('and the histogram values')
disp(' ')
disp('press a key to continue'),pause
disp(' ')
disp('do the sampling histogram with more and less samples (N)')
disp(' ')
disp('plot and comment on the similarity between the "true" histogram and')
disp('the sampled histogram for different sample sizes')

% Bayes decision theory
% we define three univariate normal distributions p1,p2,p3
% with prior probabilities P1,P2,P3
%
xmin=-10;
xmax=10;
Npdf=1000;   % number of points in grid   
dx=0.4;
% Prior probabilities

P1=0.4;
P2=0.3;
P3=1-P1-P2;
% Definitions of three normals

par1=[-2,1];    % mean value -2, variance 1
[x1,p11,x2,p12]=norm1d(par1(1),par1(2),xmin,xmax,Npdf,dx);

par2=[0,1];    % mean value 0, variance 1
[x1,p21,x2,p22]=norm1d(par2(1),par2(2),xmin,xmax,Npdf,dx);
par3=[2,1]; % mean value 2, variance 
[x1,p31,x2,p32]=norm1d(par3(1),par3(2),xmin,xmax,Npdf,dx);
figure(3),subplot(3,1,1), plot(x1,p11,'b',x1,p21,'r',x1,p31,'g'), title('class conditional densities p(x|c)')


px=(p11*P1+p21*P2+p31*P3);
figure(3), subplot(3,1,2), plot(x1,px,'m'), title('density p(x)')

P1X=p11*P1./px;
P2X=p21*P2./px;
P3X=p31*P3./px;

figure(3), subplot(3,1,3), plot(x1,P1X,'b',x1,P2X,'r',x1,P3X,'g'),title('posterior class probabilities P(c|x)')
disp('press a key to continue'),pause

% compute the confusion matrix for a given set of 
% simple  decision boundaries
%  
d12=-3.8;
d23=max(1.0,d12);
x12=[d12 d12];
y12=[0 1];
x23=[d23 d23];
y23=[0 1];
figure(4),  plot(x1,P1X,'b',x1,P2X,'r',x1,P3X,'g',x12,y12,'k',x23,y23,'k'),title('posterior class probabilities P(c|x) and decision boundaries')

indx1=find( (P1X > P2X) & (P1X > P3X));
indx2=find( (P2X > P1X) & (P2X > P3X));
indx3=find( (P3X > P2X) & (P3X > P1X));

R=probconfus(d12,d23,par1,par2,par3);
disp('The error confusion matrix is:')
disp(R)

CC=zeros(size(R));
CC=R*diag([P1 P2 P3]);
disp('Explain this alternative confusion matrix:')
disp(CC)


