function [rate,probs,class] = nc_err_frac(Wi, Wo, Inputs, Targets) 

%NC_ERR_FRAC   Neural classifier fraction of misclassified examples ... 
%  [rate,probs,class] = nc_err_frac(Wi,Wo,Inputs,Targets)
%  Calculate the fraction of erroneous classified examples, estimated
%  classes and posterior probabilities
%
%  Input:
%        Wi      : Matrix with input-to-hidden weights
%        Wo      : Matrix with hidden-to-outputs weights
%        Inputs  : Matrix with examples as rows
%        Targets : Matrix with target values as rows
%  Output:
%            rate: the fraction of erroneous classified examples
%           class: vector of estimated classes (No. of examples,1)
%           probs: matrix of posterior probabilities
%                   (No. of examples,no. of classes)
%
%  Neural Classifier, DSP IMM DTU, JL97,MWP97

  % Calculate hidden and output unit activations
  [Vj,probs] = nc_forward(Wi,Wo,Inputs);

  % Do softmax
  probs = nc_softmax(probs);


  % Choose maxima as target classes
  [v p] = max(probs');
  class = p';

  % Determine the number of examples
  exam = size(Inputs,1);
  
  rate = sum(Targets ~= class)/exam;
  
  
    
