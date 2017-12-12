function [Wi,Wo]=nr_winit(Ni,Nh,No,range,seed)

%NR_WINIT      Initial weight in neural network
%   [Wi,Wo] = NR_WINIT(Ni,Nh,No,range,seed) initialize the
%   weight in a neural network with a uniform distribution
%
%   Input:
%        Ni    : no. of input neurons
%        Nh    : no. of hidden neurons
%        No    : no. of output neurons
%        range : weight initialization uniformly over
%                [-range;range]/Ni and [-range;range]/Nh, respectively. 
%        seed  : a integer seed number, e.g., sum(clock*100)
%    Output:
%        Wi: Input-to-hidden weights
%        Wo: Hidden-to-output initial weights
%       
%    Neural Regression toolbox, DSP IMM DTU

% JL97

  rand('seed',seed);
  Wi = range*(2*rand(Nh,Ni+1)-1)/Ni;
  Wo = range*(2*rand(No,Nh+1)-1)/Nh;

