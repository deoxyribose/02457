function [n] = nc_eucnorm(dWi,dWo)

%NC_EUCNORM    Neural classifier Euclidean norm of weight vector
%  [n] = nc_eucnorm(dWi,dWo)
%  Calculates the Euclidian length of the total weight vector, 
%
%  Input:
%        dWi      :  Matrix with input-to-hidden gradient
%        dWo      :  Matrix with hidden-to-outputs gradient
%  Output:
%        n        :  Euclidian norm sqrt(|w|.^2)
%  
%  Neural classifier, DSP IMM DTU, MWP97

%  cvs: $Revision: 1.1 $

  n = sqrt( sum(sum(dWi .^ 2)) + sum(sum(dWo .^ 2)) ); 

