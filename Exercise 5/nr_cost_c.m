function [cost] = nr_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%NR_COST_C     Quadratic cost function with quadratic weight decay term
%   [cost] = NR_COST_C(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%
%   Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%   Output:
%        Cost    : Value of augmented quadratic cost function
%
%   See also NR_COST_E
%
%   Neural Regression toolbox, DSP IMM DTU

  cost = nr_cost_e(Wi,Wo,Inputs,Targets);
  cost = cost + 0.5 * ( alpha_i*sum(sum(Wi.^2)) + ...
      alpha_o*sum(sum(Wo.^2)) );

