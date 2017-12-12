function p = mvnpdf(mu,sigma,x)
% Probability Density Function of a Multivariate Normal Distribution
%
% P = mvnpdf(MU, SIGMA, X)
%
%  caculates the PDF of a multivariate normal distribution where
%  MU    is the mean (nx1)
%  SIGMA is the covariance matrix (nxn)
%  x is an nxm matrix with m points at which to calculate the PDF.
%

% (c) Karam Sidaros, August 1999.
% vektoriseret i August 2001.
  
%%%%%%%%%%%%%%%%% Check input %%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
  error('Too few input arguments');
end

[n m] = size(x);

if size(mu) ~= [n 1]
  error('MU has wrong dimension');
end
if size(sigma) ~= [n n]
  error('SIGMA has wrong dimension');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

isigma = inv(sigma);
dsigma = det(sigma);
mu = repmat(mu,1,m);

a = 1/((2*pi)^(n/2) * sqrt(dsigma));
dx = x-mu;

p = a * exp(-0.5*sum(dx.*(isigma*dx),1));

%for j = 1:m
%  p(j) = a * exp(-0.5*dx(:,j)'*isigma*dx(:,j));
%end
