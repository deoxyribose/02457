function [error] = nr_cost_e(Wi,Wo,Inputs,Targets)
%NR_COST_E     Calculate the quadratic cost function
%   [error] = NR_COST_E(Wi,Wo,Inputs,Targets) calculates the value of
%   the quadratic cost function, i.e., 0.5*(sum of squared errors)
%
%   Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%   Output:
%        error   : Value of quadratic cost function
%
%   See also NR_COST_C
%     
%   Neural Regression toolbox, DSP IMM DTU    

%   cvs: $Revision: 1.2 $

  % Calculate network outputs for all examples
  [Vj,yj] = nr_forward(Wi,Wo,Inputs);
     
  % Calculate the deviations from desired outputs 
  ej = Targets - yj;
     
  % Calculate the sum of squared errors
  error = 0.5 * sum(sum(ej .^ 2));













