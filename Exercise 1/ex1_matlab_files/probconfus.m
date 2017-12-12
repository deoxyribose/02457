function R=probconfus(d12,d23,par1,par2,par3)
% function R=probconfus(d12,d23,par1,par2,par3)
%
% computes the confusion matrix 
% R(j,k) = int_Ij p(x|k) dx
% using the "errorfunction" erf()
%
% INPUT 
%   d12 decision boundary class 1,2
%   d23 decision boundary class 2,3
%   par_k = mean, variance parameters of normal No k
% OUTPUT
%   R  3x3 confusion matrix
%
R=zeros(3);
sqrt2=sqrt(2);
R(1,1)=0.5*(erf((d12-par1(1))/sqrt2/sqrt(par1(2))) + 1);
R(1,2)=0.5*(erf((d12-par2(1))/sqrt2/sqrt(par2(2))) + 1);
R(1,3)=0.5*(erf((d12-par3(1))/sqrt2/sqrt(par3(2))) + 1);
%
R(2,1)=0.5*(erf((d23-par1(1))/sqrt2/sqrt(par1(2))) - erf((d12-par1(1))/sqrt2/sqrt(par1(2))));
R(2,2)=0.5*(erf((d23-par2(1))/sqrt2/sqrt(par2(2))) - erf((d12-par2(1))/sqrt2/sqrt(par2(2))));
R(2,3)=0.5*(erf((d23-par3(1))/sqrt2/sqrt(par3(2))) - erf((d12-par3(1))/sqrt2/sqrt(par3(2))));
%
R(3,1)=0.5*(1- erf((d23-par1(1))/sqrt2/sqrt(par1(2))));
R(3,2)=0.5*(1- erf((d23-par2(1))/sqrt2/sqrt(par2(2))));
R(3,3)=0.5*(1- erf((d23-par3(1))/sqrt2/sqrt(par3(2))));





