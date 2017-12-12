function [Wi,Wo] = nr_extract(WW,D)

%NR_EXTRACT    Extraction weight matrices from the reshaped vector
%   [Wi,Wo] = extract(WW,D) 
%
%   Input:
%        WW : the vector of the dimensions D(1)+D(2)+D(3)+D(4)
%        D  : the vector with stored dimensions of weight matrices 
%   Output:
%        Wi :  the matrix with input-to-hidden weights
%        Wo :  the matrix with hidden-to-output weights

%   cvs: $Revision: 1.1 $

  Wi = reshape(WW(1:D(1)*D(2)),D(1),D(2));
  Wo = reshape(WW(D(1)*D(2)+1:D(1)*D(2)+D(3)*D(4)),D(3),D(4));
