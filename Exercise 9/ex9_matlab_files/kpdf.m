function [estpdf]=kpdf(xtrain,h,xtest)
% Parzen window kernel pdf estimator
% 
% INPUT 
%
% xtrain    [N,D] training set
% h          variance of gaussian kernel
% xtest     [Ntest,D] test input
%
% OUTPUT
% estpdf    estimated pdf values at xtest poinst
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  COURSE 02457 (c) 2007 Lars Kai Hansen, IMM, DTU  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input1=xtrain';
input2=xtest';   % transpose data for compatibility
alf=1/h;   % use precision rather than variance
[D,N1]=size(input1);
[DD,N2]=size(input2);
if D~=DD, disp('Dimensional mismatch between train and test input'),end
%
%
input1_2=sum(input1.*input1,1);
input2_2=sum(input2.*input2,1);
%
% compute N1*N2 distance matrix
W21=(repmat(input2_2,N1,1)+repmat(input1_2',1,N2)-2*input1'*input2)';
expW21=((alf/(2*pi)).^(D/2))*exp(-0.5*alf*W21);
estpdf=sum(expW21,2)/N1;
