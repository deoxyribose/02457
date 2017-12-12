function [dWi,dWo] = nc_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)

%NC_GRADIENT   Neural classifier gradient
%  [dWi,dWo] = nc_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%  Calculate the partial derivatives of the negative log-likelihood cost.
%  wrt. the weights. Derivatives of quadratic weight decay are included.
%
%  Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%  Output:
%        dWi     :  Matrix with gradient for input weights
%        dWo     :  Matrix with gradient for output weights
%  
%  Neural classifier, DSP IMM DTU, MWP97

%  cvs: $Revision: 1.2 $

  % Determine the number of examples
  [exam inp] = size(Inputs);
 
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%

  % Calculate hidden and output unit activations
  [Vj,yj] = nc_forward(Wi,Wo,Inputs);
    
  %%%%%%%%%%%%%%%%%%%%%
  %%% BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%

  % Apply softmax
  yj = nc_softmax(yj);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Correct delta for output corresponding to the target class

  % Create indices in matrix for outputs corresp. to the correct class
  indx = (Targets-1)*exam + (1:exam)';
  
  % Subtract target value (=1) from correct class probabilities
  yj(indx) = yj(indx) - 1;
  
  % Remove dummy zeros for the last class
  [r c] = size(Wo);
  yj(:,r+1) = [];

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  % Hidden unit deltas
  delta_h =(1.0 - Vj.^2) .* (yj * Wo(:,1:c-1));
  
  % Partial derivatives for the output weights
  dWo = yj' * [Vj ones(exam,1)];

  % Partial derivatives for the input weights
  dWi = delta_h' * [Inputs ones(exam,1)];
    
  % Add derivatives of the weight decay term
  dWi = (dWi + alpha_i * Wi) / exam;
  dWo = (dWo + alpha_o * Wo) / exam;
  
  
    













