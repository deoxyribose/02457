function [Vj,phi] = nc_forward(Wi,Wo,Inputs)

%NC_FORWARD    Neural classifier forward
%  [Vj,phi] = nc_forward(Wi,Wo,Inputs)
%  Propagate examples forward through network calculating all hidden-
%  and output unit outputs. Note: There is no softmax included.
%
%  Input:
%        Wi     :  Matrix with input-to-hidden weights
%        Wo     :  Matrix with hidden-to-outputs weights
%        inputs :  Matrix with example inputs as rows
%
%  Output:
%        Vj  :  Matrix with hidden unit outputs as rows
%       phi  :  Matrix with output unit outputs as rows
%  
%  Neural classifier, DSP IMM DTU, MWP97

%  cvs: $Revision: 1.2 $


  % Determine the size of the problem
  [examples inp] = size(Inputs);  

  % Calculate hidden unit outputs for every example
  Vj = tanh([Inputs ones(examples,1) ] * Wi');
  
  % Calculate (linear) output unit outputs for every example
  phi = [Vj ones(examples,1)] * Wo';

  
  
