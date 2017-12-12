function [dWi,dWo,ddWi,ddWo] = nr_pseuhess(Wi, Wo, alpha_i, alpha_o, ...
    Inputs, Targets)
%NR_PSEUHESS   Pseudo Hessian elements and the partial derivatives.
%   [dWi,dWo,ddWi,ddWo] = NR_PSEUHESS(Wi,Wo,alpha_i,alpha_o,Inputs,Targets) 
%   calculates the pseudo Hessian elements AND the partial derivatives
%   of the quadratic cost function  wrt. the weights. Derivatives of
%   quadratic weight decay are included.
%
%   Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%   Output:
%        dWi     :  Matrix with gradient for input weights
%        dWo     :  Matrix with gradient for output weights
%        ddWi    :  Matrix with pseudo Hessian for input w.
%        ddWo    :  Matrix with pseudo Hessian for output w.
%
%   Neural Regression toolbox, DSP IMM DTU


  [exam inp] = size(Inputs);    % Determine the number of examples

  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%

  % Calculate hidden and output unit activations
  [Vj,yj] = nr_forward(Wi,Wo,Inputs);
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% PSEUDO HESSIAN BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Calculate derivative of OUTPUT UNIT OUTPUTS 
  % by backpropagating ones instead of errors from 
  % desired outputs
  
  [r c] = size(Wo);
  delta_o = ones(exam,r);
  delta_h =(1.0 - Vj.^2) .* (delta_o * Wo(:,1:c-1));

  hb = [Vj ones(exam,1)];       % Hidden unit outputs and bias
    
  % Pseudo Hessian elements for the output weights
  ddWo = delta_o' * (hb .^ 2);

  hi = [Inputs ones(exam,1)];   % External inputs and bias

  % Pseudo Hessian elements for the input weights
  ddWi = (delta_h .^ 2)' * (hi .^ 2);
   
  % Add second derivatives of weight decay term
  ddWi = ddWi + alpha_i;
  ddWo = ddWo + alpha_o;    

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% GRADIENT BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Calculate derivative of 
  % by backpropagating the errors from the
  % desired outputs

  % Output unit deltas
  delta_o = -(Targets - yj);
  
   % Hidden unit deltas
  delta_h =(1.0 - Vj.^2) .* (delta_o * Wo(:,1:c-1));
  
  % Partial derivatives for the output weights
  dWo = delta_o' * [Vj ones(exam,1)];

  % Partial derivatives for the input weights
  dWi = delta_h' * [Inputs ones(exam,1)];
  
  % Add derivatives of the weight decay term
  dWi = dWi + alpha_i * Wi;
  dWo = dWo + alpha_o * Wo;
  













