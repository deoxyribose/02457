function [errors,probs,class] = nc_cl_error(Wi, Wo, Inputs, Targets)

%NC_CL_ERROR   Neural classifier 
%  [errors,probs,class] = cl_error(Wi,Wo,Inputs,Targets)
%  Calculate number of erroneous classified examples, estimated classes
%  and posterior probabilities
%
%  Input:
%        Wi      : Matrix with input-to-hidden weights
%        Wo      : Matrix with hidden-to-outputs weights
%        Inputs  : Matrix with examples as rows
%        Targets : Matrix with target values as rows
%  Output:
%          errors: the no. erroneous classified examples
%           class: vector of estimated classes (No. of examples,1)
%           probs: matrix of posterior probabilities
%                  (No. of examples,no. of classes)
%
%  Neural classifier, DSP IMM DTU, JL97,MWP97

%  cvs: $Revision: 1.1 $


  % Calculate hidden and output unit activations
  [Vj,probs] = nc_forward(Wi,Wo,Inputs);

  % Compute softmax
  probs = nc_softmax(probs);
 
  % Choose maxima as target classes
  [v p] = max(probs');
  class = p';
  
  errors = sum (Targets ~= class);
  
  
    








