function [ mu,Sigma] = posterior_update_linear(x,t,mu,Sigma,alpha,beta)
% fixed prior, noise level update using the forward algorithm
% mu        mean vector (d+1) x 1 
% Sigma     covariance matrix  (d+1) x (d+1)
%
  dp1=size(Sigma,1);
  iSigma=pinv(Sigma + (1/alpha)*eye(dp1));
  Sigma= pinv(beta*x*x' + iSigma);
  mu=Sigma*(beta*t*x + iSigma*mu);
end

