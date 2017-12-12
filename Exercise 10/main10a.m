% %  main9.a  MATLAB main program for Kernel pdf estimation
%   Course 02457, November 2007, LKH
%  
clear
%
N1=50;          % Number of data points in d=2 data 
N2=50;
sig=1.5;sig2=sig^2;
%
centers=zeros(2,2);
centers(1,1)=1;
centers(2,1)=-1;
centers(1,2)=-1;
centers(2,2)=1;
width=sqrt(0.1);       % std of simulation normal distribution
%
% Normal distributed data for training set clusters
train1=centers(:,1)*ones(1,N1) + width*randn(2,N1);
train2=centers(:,2)*ones(1,N2) + width*randn(2,N2);
train=[train1,train2];
%
subplot(2,2,1)
plot(train1(1,:),train1(2,:),'r.',train2(1,:),train2(2,:),'b.')
grid
dists=gp_dist(train,train);
K=exp( - dists/(2*sig2));
%
subplot(2,2,2)
imagesc(K,[0 1]), colormap('gray'),colorbar
%
subplot(2,2,3)
width=sqrt(2);       % std of simulation normal distribution
%);
% Normal distributed data for training set clusters
train1=centers(:,1)*ones(1,N1) + width*randn(2,N1);
train2=centers(:,2)*ones(1,N2) + width*randn(2,N2);
train=[train1,train2];
plot(train1(1,:),train1(2,:),'r.',train2(1,:),train2(2,:),'b.')
grid
dists=gp_dist(train,train);
K=exp( - dists/(2*sig2));
subplot(2,2,4)
imagesc(K,[0 1]), colormap('gray'),colorbar








