function [dim] = nr_dimen(Wi,Wo)
%NR_DIMEN      Number of non-zero-weigths 
%   [dim] = NR_DIMEN(Wi,Wo) calculates the number of non-zero weights
%   in the network, i.e. the dimension of the total weight vector 
%
%   Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%   Output:
%        dim     :  Number of non-zero weights 
%
%   Neural Regression toolbox, DSP IMM DTU


  dim = sum(sum(Wi~=0));        % Number of non-zero input weights
  
  dim = dim + sum(sum(Wo~=0));  % Add number of non-zero output weights
  
