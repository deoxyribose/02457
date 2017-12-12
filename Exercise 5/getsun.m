function [tr_i,tr_t,te_i,te_t] = getsun(d)
%
% Create input and output data from a 'teacher' network. The outputs
% are contaminated with additive white noise.
% Inputs:
%         d :  Sunspot lagspace dimension
%         ptrain :  Number of training examples
%         ptest  :  Number of test examples
% Outputs:
%         tr_i, te_i :  Inputs for training & test set
%         tr_t, te_t :  Target values
S = load('sp.dat'); % Load sunspot data-set
year = S(:,1);  
S = S(:,2);
var = std(S).^2;   % the total signal variance for normalization
last_train = 221-d;                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create lag space matrix 
N = length(S)-d;
T = S(d+1:length(S));
X = ones(N,1);
for a = 1:N
  X(a,1:d) = S(a:a+d-1)';
end
%Training set
tr_i=X(1:last_train,:);
te_i=X((last_train+1):N,:);
tr_t=T(1:last_train);
te_t=T((last_train +1):N);


