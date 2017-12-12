%  MATLAB main program for neural network training and pruning
%
%  The network is initially trained using gradient methode.
%  

clear
Ni = 4;                   % Number of external inputs
Nh = 5;                   % Number of hidden units
No = 1;                   % Number of output units
alpha_i = 0.0;           % Input weight decay
alpha_o = 0.0;           % Output weight decay
max_iter=500;             % maximum number of iterations
eta=0.001;                % gradient descent parameter
t_Nh = 2;                 % Number of hidden units in TEACHER net
noise = 1.0;              % Relative amplitude of additive noise
ptrain = 100;             % Number of training examples
ptest = 100;              % Number of test examples
I_gr = 1;                 % Initial max. gradient iterations
range =0;               % Initial weight range                

randn('seed',sum(100*clock));


% First, get some data...
[train_inp,train_tar,test_inp,test_tar] = getdata(Ni,t_Nh,No,ptrain,ptest,noise);

% Initialize network weights
Wi = range * randn(Nh,Ni+1);
Wo = range * randn(No,Nh+1);
iter = 1;
while iter < max_iter
  [dWi,dWo] =  gradient(Wi,Wo,alpha_i,alpha_o,train_inp,train_tar)
  % Update weights
  Wi = Wi - eta * dWi
  Wo = Wo - eta * dWo
  
  % Calc error and gradient
  Gradient(iter)= two_norm(dWi,dWo);
  Etrain(iter) = 2*cost_e(Wi,Wo,train_inp,train_tar)/ptrain;
  Etest(iter) = 2*cost_e(Wi,Wo,test_inp,test_tar)/ptest;
  iter = iter + 1;
  
  % Plot Costfunction
  figure(1)
  semilogy(1:length(Etrain),Etrain,1:length(Etest),Etest,'g:')
  legend('Train','Test')
  ylabel('cost')
  xlabel('iterations')
  drawnow 
end
clear Etrain
clear Etest


