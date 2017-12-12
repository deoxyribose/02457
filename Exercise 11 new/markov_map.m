function a=markov_map(seq,K,alf)
%
% function a=markov(seq,K);
%   a(j,k)= prob ( x(t)=k and x(t-1) = j)
% alf = alpha-1
%
ht=seq(1:(end-1)) + K*(seq(2:end)-1);
n=hist(ht,1:(K*K));
a=reshape(n,K,K)+alf;
asum=sum(a,2);
a=a.*repmat(1./asum,1,K);

