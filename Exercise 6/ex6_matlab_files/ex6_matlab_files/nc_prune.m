function [Wi_new,Wo_new] = nc_prune(Wi, Wo, alpha_i, alpha_o, Inputs, ...
    Targets, kills)

%NC_PRUNE      Neural classifier pruning
%  [Wi_new,Wo_new] = prune(Wi,Wo,alpha_i,alpha_o,Inputs,Targets,kills)
%  This function prunes a number of weights away using Optimal Brain
%  Damage. In the current implementation, the gradient is assumed to
%  be non-zero. 
%
%  Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        kills   :  Number of weights to eliminate
%  Output:
%        Wi_new  :  Matrix with reduced input-to-hidden weights
%        Wo_new  :  Matrix with reduced hidden-to-outputs weights
%  
%  Neural classifier, DSP IMM DTU, JL97, MWP97

%  cvs: $Revision: 1.1 $

  % Calculate second derivatives WITHOUT weight decay term
  [dWi,dWo,ddWi,ddWo] =  nc_pseuhess(Wi,Wo,0,0,Inputs,Targets);
 
  % Calculate saliencies for the input weights INCLUDING gradient term
  Sal_input = (alpha_i + 0.5 * ddWi .^ 2) .* (Wi .^ 2) - (dWi .* Wi); 

  % Calculate saliencies for the output weights INCLUDING gradient term
  Sal_output = (alpha_o + 0.5 * ddWo .^ 2) .* (Wo .^ 2) - (dWo .* Wo);

  % Set saliencies for 'dead' weights to LARGE values
  Sal_input = Sal_input + (realmax * (Wi==0));  
  Sal_output = Sal_output + (realmax * (Wo==0));  

  [n1,n2]=size(Wi);
  mi=n1*n2;
  [n1,n2]=size(Wo);
  mo=n1*n2;


  [wsort,idx] = sort([reshape(Sal_input,mi,1);reshape(Sal_output,mo,1)]);
  idxkill=idx(1:kills);
  idxwi=idxkill<=mi;
  idxwo=idxkill>mi;
  if any(idxwi)~=0
    Wi(idxkill(idxwi))=zeros(sum(idxwi==1),1);
  end
  if any(idxwo)~=0
    Wo(idxkill(idxwo)-mi)=zeros(sum(idxwo==1),1);
  end
  Wi_new=Wi;
  Wo_new=Wo;
    
    
    












