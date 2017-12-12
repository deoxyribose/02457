function [cost] = nc_cost_c(Wi, Wo, alpha_i, alpha_o, Inputs, ...
    Targets) 

%NC_COST_C     Neural classifier costfunction with regularization
%  [cost] = nc_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%  Calculate the value of the negative log-likelihood cost function,
%  augmented by quadratic weight decay term
%
%  Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%  Output:
%        Cost    : Value of augmented negative log-likelihood cost function
%  
%  Neural classifier, DSP IMM DTU, MWP97

%  cvs: $Revision: 1.1 $

  [exam inp] = size(Inputs);    % Determine the number of examples
  cost = nc_cost_e(Wi,Wo,Inputs,Targets);
  cost = cost + 0.5 * ( alpha_i*sum(sum(Wi.^2)) + ...
      alpha_o*sum(sum(Wo.^2)) ) / exam; 

