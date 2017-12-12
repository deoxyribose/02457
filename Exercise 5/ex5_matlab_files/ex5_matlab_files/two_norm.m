function [n] = two_norm(dWi,dWo)
%
% Calculates the Euclidian length of the total weight vector, 
% i.e. the 2-norm.
%
% Input:
%        dWi      :  Matrix with input-to-hidden gradient
%        dWo      :  Matrix with hidden-to-outputs gradient
% Output:
%        n        :  2-norm of total gradient

  n = sqrt( sum(sum(dWi .^ 2)) + sum(sum(dWo .^ 2)) ); 


