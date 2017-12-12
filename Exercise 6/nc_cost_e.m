function [error] = nc_cost_e(Wi,Wo,Inputs,Targets)

%NC_COST_E     Neural classifier costfunction (without regularization) 
%  [error] = nc_cost_e(Wi,Wo,Inputs,Targets)
%  Calculate the value of the negative log likelihood cost
%
%  Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        Inputs  :  Matrix with input features as rows
%        Targets :  Column vector  with target class as elements
%  Output:
%        error   :  Value of negative log likelihood
%  
%  Neural classifier, DSP IMM DTU, JL97, MWP97

%  cvs: $Revision: 1.2 $

  % Calculate network outputs for all examples
  [Vj,yj] = nc_forward(Wi,Wo,Inputs);
  
  % Expand output unit outputs with zeros for the last class
  yj = [yj zeros(max(size(yj)),1)];

  [exam,c]=size(yj);

  % Offsets for numerical trick
  offsets = max(yj')';
  offsets_expand=kron(ones(1,c),offsets);

  % Determine column vector with outputs corresponding to 
  % the correct classes
  for i=1:exam
    correct_out(i)=yj(i,Targets(i));
  end
  correct_out = correct_out'; 

 
  % Calculate individual likelihoods using numerical trick
  likelihoods = offsets + log( sum( exp(yj-offsets_expand)' )' ) - ...
      correct_out; 

  % Calculate mean sum of likelihoods
  error = sum(likelihoods)/exam;

