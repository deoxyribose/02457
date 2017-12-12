% Matlab exercise main9b.m 
% Demonstration of K-nearest neighbor classification
% for the Pima diabetes diagnosis
%
load pima
Kmax=120;
figure(1)
X_tr=X_tr(:,[1,2,3,4,5,6,7]);
X_te=X_te(:,[1,2,3,4,5,6,7]);
%
Ntrain=size(X_tr,1);
Ntest=size(X_te,1);
X_te=X_te-repmat(mean(X_tr,1),Ntest,1);

X_tr=X_tr-repmat(mean(X_tr,1),Ntrain,1);
X_te=X_te./repmat(std(X_tr,[],1),Ntest,1);
X_tr=X_tr./repmat(std(X_tr,[],1),Ntrain,1);
%
%
[looclass1,class2,Kopt,error_array]=multiclass_knn(X_tr',y_tr',X_te',Kmax);
plot(1:Kmax,error_array,'ro')
axis([0 Kmax 0 0.5])
grid
title('LOO ERROR')
Errtest=sum(class2~=y_te')/size(y_te,1)
xlabel('Number of nearest neigbors in voting')
ylabel('Error rate'),setfig

