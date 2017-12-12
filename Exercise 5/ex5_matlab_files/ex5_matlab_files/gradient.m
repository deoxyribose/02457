function [dWi,dWo] = gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%
% Calculate the partial derivatives of the quadratic cost function
% wrt. the weights. Derivatives of quadratic weight decay are included.
%
% Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
% Output:
%        dWi     :  Matrix with gradient for input weights
%        dWo     :  Matrix with gradient for output weights


% Determine the number of examples
[exam inp] = size(Inputs);

%%%%%%%%%%%%%%%%%%%%
%%% FORWARD PASS %%%
%%%%%%%%%%%%%%%%%%%%

% Calculate hidden and output unit activations
[Vj,yj] = forward(Wi,Wo,Inputs);

%%%%%%%%%%%%%%%%%%%%%
%%% BACKWARD PASS %%%
%%%%%%%%%%%%%%%%%%%%%

% Calculate derivative of 
% by backpropagating the errors from the
% desired outputs

% Output unit deltas
delta_o = -(Targets - yj);

% Hidden unit deltas
[r c] = size(Wo);
delta_h =(1.0 - Vj.^2) .* (delta_o * Wo(:,1:c-1));

% Partial derivatives for the output weights
dWo = delta_o' * [Vj ones(exam,1)];

% Partial derivatives for the input weights
dWi = delta_h' * [Inputs ones(exam,1)];

% Add derivatives of the weight decay term
dWi = dWi + alpha_i * Wi;
dWo = dWo + alpha_o * Wo;



