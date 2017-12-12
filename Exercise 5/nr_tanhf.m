function y=tanhf(x)

%NR_TANHF      Fast hypobolic tangent
%   y=tanhf(x) calculates the fast hypobolic tangent:
%   y=1 - 2./(exp(2*x)+1);
%
%   Neural Regression toolbox, DSP IMM DTU

  y=sign(x).*(2./(1+exp(-2*abs(x)))-1);