function [dWi,dWo,ddWi,ddWo] = nc_pseuhess(Wi, Wo, alpha_i, alpha_o, ...
    Inputs, Targets)

%NC_PSEUHESS   Neural classifier pseudo-Hessian
%  [dWi,dWo,ddWi,ddWo] = pseuhess(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%  Calculate the pseudo Hessian (diagonal) elements AND the partial
%  derivatives of the negative log-likelihood cost function  wrt. the
%  weights. Derivatives of quadratic weight decay are included. 
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
%        ddWi    :  Matrix with pseudo Hessian for input weights
%        ddWo    :  Matrix with pseudo Hessian for output weights
%  
%  Neural clssifier, DSP IMM DTU, MWP97

%  cvs: $Revision: 1.1 $

  [exam inp] = size(Inputs);    % Determine the number of examples

  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%

  % Calculate hidden and output unit activations
  [Vj,yj] = nc_forward(Wi,Wo,Inputs);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Compute softmax probabilities
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  yj = nc_softmax(yj);

    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% PSEUDO HESSIAN BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  [r c] = size(Wo);

  % Output unit deltas
  delta_o = yj(:,1:r) .* (1 - yj(:,1:r));
  
  % Hidden unit deltas
  delta_h =((1.0 - Vj.^2) .^2);
  delta_h = delta_h  .* ( yj(:,1:r) * (Wo(:,1:c-1).^2) - (yj(:,1:r) * ...
      Wo(:,1:c-1)).^2 ); 

  hb = [Vj ones(exam,1)];       % Hidden unit outputs and bias
    
  % Pseudo Hessian elements for the output weights
  ddWo = delta_o' * (hb .^ 2);

  hi = [Inputs ones(exam,1)];   % External inputs and bias

  % Pseudo Hessian elements for the input weights
  ddWi = delta_h' * (hi .^ 2);
   
  % Add second derivatives of weight decay term
  ddWi = (ddWi + alpha_i) / exam;
  ddWo = (ddWo + alpha_o) / exam;    

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% GRADIENT BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Correct delta for output corresponding to the target class

  % Create indices in matrix for outputs corresp. to the correct class
  indx = (Targets-1)*exam + (1:exam)';
  
  % Subtract target value (=1) from correct class probabilities
  yj(indx) = yj(indx) - 1;
  
  % Remove dummy zeros for the last class
  yj(:,r+1) = [];

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  % Hidden unit deltas
  delta_h = (1.0 - Vj.^2) .* (yj * Wo(:,1:c-1));
  
  % Partial derivatives for the output weights
  dWo = yj' * [Vj ones(exam,1)];

  % Partial derivatives for the input weights
  dWi = delta_h' * [Inputs ones(exam,1)];
    
  % Add derivatives of the weight decay term
  dWi = (dWi + alpha_i * Wi) / exam;
  dWo = (dWo + alpha_o * Wo) / exam;
  
  
    

