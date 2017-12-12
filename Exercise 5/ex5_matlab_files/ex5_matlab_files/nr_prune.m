function [Wi_new,Wo_new] = nr_prune(Wi,Wo,alpha_i,alpha_o,Inputs,Targets,kills) 
%NR_PRUNE      Prune weights with Optimal Brain Damage
%   [Wi_new,Wo_new] = nr_prune(Wi,Wo,alpha_i,alpha_o,Inputs,Targets,kills)
%   prunes a number of weights away using Optimal Brain Damage
%
%   Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        kills   :  Number of weights to eliminate
%   Output:
%        Wi_new  :  Matrix with reduced input-to-hidden weights
%        Wo_new  :  Matrix with reduced hidden-to-outputs weights
%
%   See also NR_TRAIN, NR_PSEUHESS
%
%   Neural Regression toolbox, DSP IMM DTU

  % Calculate second derivatives WITHOUT weight decay term
  [dWi,dWo,ddWi,ddWo] =  nr_pseuhess(Wi,Wo,0,0,Inputs,Targets);
 
  % Calculate saliencies for the input weights
  Sal_input = (alpha_i + 0.5 * ddWi .^ 2) .* (Wi .^ 2); 

  % Calculate saliencies for the output weights
  Sal_output = (alpha_o + 0.5 * ddWo .^ 2) .* (Wo .^ 2);
  
  % Eliminate 'kills' number of weights with the smallest saliencies
  for i=1:kills
  
    % Set saliencies for 'dead' weights to LARGE values
    Sal_input = Sal_input + (realmax * (Wi==0));  
    Sal_output = Sal_output + (realmax * (Wo==0));  
   
    % Determine smallest saliency for the input weights
    min_i = min(min(Sal_input));
    
    % Determine smallest saliency for the output weights
    min_o = min(min(Sal_output));
    
    if min_i < min_o
    
      % Get coordinates for input weight with smallest saliency
      if max(size(Wi(:,1))) > 1
        % matrix 
        [val_col ind_row] = min(Sal_input);
        [min_val c] = min(val_col);
        r = ind_row(c);
      else
        % vector
        [min_val c] = min(Sal_input);
        r = 1;
      end;  
      
      % Kill weight       
      Wi(r,c) = 0; 
      
    else
    
      % Get coordinates for output weight with smallest saliency
      if max(size(Wo(:,1))) > 1
        % matrix 
        [val_col ind_row] = min(Sal_output);
        [min_val c] = min(val_col);
        r = ind_row(c);
      else
        % vector
        [min_val c] = min(Sal_output);
        r = 1;
      end;  

      % Kill weight       
      Wo(r,c) = 0;
      
    end;

  end;        % 'kills' number of weights eliminated
  
  Wi_new = Wi;
  Wo_new = Wo;
  
    
    






