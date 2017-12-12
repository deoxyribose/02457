function y=tanhf(x)

% Fast tanh y=tanhf(x)
%y=1 - 2./(exp(2*x)+1);
y=sign(x).*(2./(1+exp(-2*abs(x)))-1);