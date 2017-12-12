function [dim] = nc_dimen(Wi,Wo)

%NC_DIMEN      Neural classifier number of non-zero weights
%  [dim] = dimen(Wi,Wo)
%  Calculates the number of non-zero weights in the 
%  network, i.e. the dimension of the total weight vector
%
%  Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%  Output:
%        dim     :  Number of non-zero weights 
%  
%  Neural Classifier, DSP IMM DTU, MWP97

%  cvs: $Revision: 1.1 $

  dim = sum(sum(Wi~=0));        % Number of non-zero input weights
  
  dim = dim + sum(sum(Wo~=0));  % Add number of non-zero output weights
  
