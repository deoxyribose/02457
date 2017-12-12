function [cost] = cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%
% Calculate the value of the quadratic cost function,
% augmented by quadratic weight decay term
%
% Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
% Output:
%        Cost    : Value of augmented quadratic cost function

  cost = cost_e(Wi,Wo,Inputs,Targets);
  cost = cost + 0.5 * ( alpha_i*sum(sum(Wi.^2)) + alpha_o*sum(sum(Wo.^2)) );

