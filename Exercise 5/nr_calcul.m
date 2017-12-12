function [f,df] = nr_calcul(W,Dim,alpha_i, alpha_o,Inputs,Targets);

%NR_CALCUL     Calculates the cost function and the gradient
%   [f,df] = nr_calcul(X, Dim, alpha_i, alpha_o, Inputs, Targets);
%   Calculates the cost function value f and the gradient df for
%   neural network the function is operating on the vector W created
%   by reshaping matrices Wi and Wo in Dim vector the dimensions of
%   those matrices are stored.

%   cvs: $Revision: 1.1 $

  [Wi,Wo] = nr_extract(W,Dim);
  [dWi,dWo] = nr_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
  f = nr_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
  df = [reshape(dWi,1,Dim(1)*Dim(2))  reshape(dWo,1,Dim(3)*Dim(4))];
