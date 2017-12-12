function r = randbin(p,m,n)
% produces a matrix with random binomial values (0 or 1)
%
% R = randbin(P,M,N)
%   gives a MxN matrix with randomly placed 0's and 1's 
%   where P is the probability of finding a 1 in any place.
%
% If N i omitted, a vector of length M is returned
%

% (c) Karam Sidaros. August, 1999.

if nargin < 2
  error('Not enough input arguments');
end

if nargin < 3
  n = 1;  
end

r = rand(m,n);
r = r <= p;

