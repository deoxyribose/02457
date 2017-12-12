function [y,indx_array]=knn_regress_demo(x,t,K,xtest,alpha)
% nearest neighbor regression 
% x is N,D array of inputs
% t i N,1  targets
% y output estimates at xtest
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 02457 (c) 2007 LKH, IMM, DTU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,D]=size(x);
[Ntest,D]=size(xtest);
indx_array=zeros(Ntest,K);
%
X_train=ones(N,D+1);
X_train(:,1:D)=x;
X_test=ones(Ntest,D+1);
X_test(:,1:D)=xtest;
%
y=zeros(Ntest,1);
for j=1:Ntest,
    if rem(j,20)==0, disp([' n = ',int2str(j),' of ',int2str(Ntest)]),end
    delta=(x-repmat(xtest(j,:),N,1));
    dist=sum(delta.*delta,2);
    [dummy indx]=sort(dist);
    X=X_train(indx(1:K),:);
    w=inv(X'*X+alpha*eye(D+1))*X'*t(indx(1:K));
    y(j)=X_test(j,:)*w;
    indx_array(j,:)=indx(1:K)';
end,  
%disp('o')
