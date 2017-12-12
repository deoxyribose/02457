% main7a.m  demonstration of k-means on 2D data
% 
% (c) Lars Kai Hansen (1999)
% revised by Carsten Stahlhut (2009)
%
  Ntrain=120;                  % number of training examples
  noise=0.06;                  % width of true clusters
  K=4;                         % Number of clusters  
  D=2;                         % Dimension of data
  nits=10;                     % Number of k-means iteration
  initial_width=0.3;           % Width of initial distribution of clusters
  close all
%
%
% getdata
[xtrain,xtest]=getdata(Ntrain,0,noise);

% initial K clusters i 2D: Initial width controls the initial clusters
y=ones(K,1)*mean(xtrain)+initial_width*randn(K,D);

% compute the square of the data vectors
x2train=ones(K,1)*sum((xtrain.*xtrain)');

% plot data points
figure(1), h_train = plot(xtrain(:,1),xtrain(:,2),'.');
hold on

% Start k-means iterations
for t=1:nits,
  % computer the distance between clusters and datavectors (K*N matrix)
  dist=sum((y.*y)')'*ones(1,Ntrain) + x2train -2*y*xtrain';   
  % Find index of closest cluster
  [a,index]=min(dist); 
  % Resestimate the cluster center
  for k=1:K
    indexk=find(index==k);
    if sum(indexk) > 0,
       y(k,:)=mean(xtrain(indexk,:));
    end
    % plot the cluster centers
    h_ite = plot(y(k,1),y(k,2),'r*'); text(y(k,1),y(k,2),[int2str(k),'-',int2str(t)]),drawnow
  end
end   %end kmeans
legend([h_train, h_ite],'train','\mu_k^{ite}')

% Plot resulting clustering
figure(2), 
for n=1:Ntrain,
   text(xtrain(n,1),xtrain(n,2),int2str(index(n))),
end   
  axis([0 1 0 1])
   hold on,plot(y(:,1),y(:,2),'r*')


