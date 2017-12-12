function kern = svm_kernel_rbf(u,v,sigma)
%Radial basis function kernel

kern = exp(-(1/(2*sigma^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))-2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
