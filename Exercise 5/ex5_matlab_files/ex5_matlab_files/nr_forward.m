function [Vj,yj] = nr_forward(Wi,Wo,Inputs)
%NR_FORWARD    Propagate example forward through the network
%   [Vj,yj] = forward(Wi,Wo,Inputs) propagates examples forward
%   through network calculating all hidden- and output unit outputs
%
%   Input:
%        Wi     :  Matrix with input-to-hidden weights
%        Wo     :  Matrix with hidden-to-outputs weights
%        inputs :  Matrix with example inputs as rows
%
%   Output:
%        Vj  :  Matrix with hidden unit outputs as rows
%        yj  :  Vector with output unit outputs as rows
%
%   Neural Regression toolbox, DSP IMM DTU


  % Determine the size of the problem
  [examples inp] = size(Inputs);  

  % Calculate hidden unit outputs for every example

  Vj = nr_tanhf([Inputs ones(examples,1) ] * Wi');
  
  % Calculate (linear) output unit outputs for every example
  yj = [Vj ones(examples,1)] * Wo';


