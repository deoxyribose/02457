function [error] = cost_e(Wi,Wo,Inputs,Targets)
%
% Calculate the value of the quadratic cost function,
% i.e. 0.5*(sum of squared errors)
%
% Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
% Output:
%        error   : Value of quadratic cost function

    
  % Calculate network outputs for all examples
  [Vj,yj] = forward(Wi,Wo,Inputs);
     
  % Calculate the deviations from desired outputs 
  ej = Targets - yj;
     
  % Calculate the sum of squared errors
   error = 0.5 * sum(sum(ej .^ 2));

