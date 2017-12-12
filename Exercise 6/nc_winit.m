function [Wi,Wo] = nc_winit(Ni,Nh,No,range,seed)

%NC_WINIT      Neural classifier weight initialization 
%  [Wi,Wo] = nc_winit(Ni,Nh,No,range,range,seed)
%  Uniform weight initialization
%  Input:
%      Ni: no. of input neurons
%      Nh: no. of hidden neurons
%      No: no. of output neurons
%   range: weight initialization uniformly over [-range;range]/Ni
%          and [-range;range]/Nh, respectively.
%    seed: a integer seed number, e.g., sum(clock*100)
%  Output:
%      Wi: Input-to-hidden weights
%      Wo: Hidden-to-output initial weights
%                                  
%  Neural classifier, DSP IMM DTU, JL97

%  cvs: $Revision: 1.1 $

  rand('seed',seed);
  Wi = range*(2*rand(Nh,Ni+1)-1)/Ni;
  Wo = range*(2*rand(No,Nh+1)-1)/Nh;

