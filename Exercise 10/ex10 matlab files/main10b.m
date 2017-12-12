%  main10b  MATLAB main program for Kernel pdf estimation
%   Course 02457, November 2012, LKH
%   Inspection of the kernel matrix for the sunspots data
%
clear, close all
%
%
d=2;  % input dimension of sunspot prediction 
sig2=1.0;
[train_x,train_t,test_x,test_t] = getsun(d);
Ntrain=length(train_t);
Ntest=length(test_t);
% plot the data as 3-dimensional cloud
figure(1)
plot3(train_x(:,1),train_x(:,2),train_t,'.')
grid
%
dist_train=gp_dist(train_x',train_x');
Ktrain=exp( - dist_train/(2*sig2) );
dist_train_test=gp_dist(train_x',test_x');
Ktrain_test=exp( - dist_train_test/(2*sig2) );
figure(2)
subplot(2,2,1), 
imagesc(1700+(1:Ntrain),1700+(1:Ntrain),Ktrain,[0 1]),colormap('gray'),colorbar
subplot(2,2,2),
plot(1700+d+(1:Ntrain),train_x(:,1),'b',1700+d+(1:Ntrain),train_x(:,2),'r',1700+d+(1:Ntrain),train_x(:,1),'bo',1700+d+(1:Ntrain),train_x(:,2),'ro'),grid
subplot(2,2,3),
imagesc(1700+d+(1:Ntrain),1920+d+(1:Ntest),Ktrain_test',[0 1]),colormap('gray'),colorbar
subplot(2,2,4)
plot(1920+d+(1:Ntest),test_x(:,1),'b',1920+d+(1:Ntest),test_x(:,2),'r',1920+d+(1:Ntest),test_x(:,1),'bo',1920+d+(1:Ntest),test_x(:,2),'ro'), grid





