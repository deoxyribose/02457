function [probs] = nc_cl_probs(Wi, Wo, Inputs)

%NC_CL_PROBS   Neural classifier posterior prob. of example
%  [probs] = nc_cl_probs(Wi,Wo,Inputs)
%  Calculate posterior probabilities for each example
%
%  Input:
%        Wi      : Matrix with input-to-hidden weights
%        Wo      : Matrix with hidden-to-outputs weights
%        Inputs  : Matrix with examples as rows
%  Output:
%           probs: matrix of posterior probabilities
%                  (No. of examples,no. of classes)
%
%  Neural classifier, DSP IMM DTU, JL97,MWP97

  % Calculate hidden and output unit activations
  [Vj,probs] = nc_forward(Wi,Wo,Inputs);

  % Do softmax
  probs = nc_softmax(probs);

  
    
