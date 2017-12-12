%NR_NETPRUN    Main program for neural network training and pruning 
%   The network is initially trained using gradient descent and pseudo
%   Gauss-Newton, using linesearch to determine step lengths. 
%   Afterwards, pruning is performed using the Optimal Brain
%   Damage method.
%
%   Neural Regression toolbox, DSP IMM DTU

%   Programmed October 1995 by * Morten With Pedersen *

clear
Ni = 4;                   % Number of external inputs
Nh = 5;                   % Number of hidden units
No = 1;                   % Number of output units
alpha_i = 0.01;           % Input weight decay
alpha_o = 0.01;           % Output weight decay

t_Nh = 2;                 % Number of hidden units in TEACHER net
noise = 1.0;              % Relative amplitude of additive noise
ptrain = 100;             % Number of training examples
ptest = 100;              % Number of test examples

I_gr = 10;                % Initial max. gradient iterations
I_psgn = 250;             % Initial max. pseudo Gauss-Newt iterations
P_gr = 0;                 % Max. gradient it. between pruning sessions
P_psgn = 50;              % Max. pseudo GN between pruning sessions
greps = 1e-3;             % Gradient norm stopping criteria
kills = 1;                % Number of weights to prune away at a time
min_dim = 2;              % Minimum dimension of network


  randn('seed',sum(100*clock));

  % First, get some data
  [train_inp, train_tar, test_inp, test_tar] = nr_getdata(Ni, t_Nh, ...
      No, ptrain, ptest, noise);

  % Initialize network weights
  Wi = randn(Nh,Ni+1)/2;
  Wo = randn(No,Nh+1)/2;

  % Perform initial training of the net
  [Wi,Wo] = nr_train(Wi, Wo, alpha_i, alpha_o, train_inp, train_tar, ...
      I_gr, I_psgn, greps);

  dim = nr_dimen(Wi,Wo);
  iter = 1;
  dimvec(iter) = dim;
  Etrain(iter) = nr_cost_e(Wi,Wo,train_inp,train_tar);
  Etest(iter) = nr_cost_e(Wi,Wo,test_inp,test_tar);
  iter = iter + 1;

  while dim > min_dim 
    % Prune weights away
    [Wi,Wo] = nr_prune(Wi,Wo,alpha_i,alpha_o,train_inp,train_tar,kills);

    [Wi,Wo] = nr_train(Wi, Wo, alpha_i, alpha_o, train_inp, ...
	train_tar, P_gr, P_psgn, greps);

    dim = nr_dimen(Wi,Wo);     % Determine new dimension
    dimvec(iter) = dim;
    Etrain(iter) = nr_cost_e(Wi,Wo,train_inp,train_tar);
    Etest(iter) = nr_cost_e(Wi,Wo,test_inp,test_tar);
    iter = iter + 1;
  
    figure(3)
    semilogy(dimvec,Etrain)
    hold on
    semilogy(dimvec,Etest,':')
    hold off
    drawnow
    title('Cost function (without weight decay term)')
    xlabel('Number of parameters (weights)')
    ylabel('cost function')
    legend('Training set', 'Test set')

  end















