
%MAIN7C        Neural classifier, Gaussian
%
%  DSP IMM DTU, Finn

%  cvs: $Revision: 1.1 $

  
  clear 
  close all
  randn('seed',0) 

  Ni = 1;                   % Number of external inputs
  Nh = 4;                   % Number of hidden units
  No = 1;                   % Number of output units
  
  % Weight initialization
  seed=sum(100*clock);
  range=0.3;
  
  alpha_i = 1e-3;           % Input weight decay
  alpha_o = 1e-3;           % Output weight decay
  
  I_gr = 5;                 % Initial max. gradient iterations
  I_psgn = 30;              % Initial max. pseudo Gauss-Newt iterations
  P_gr = 0;                 % Max. gradient it. between pruning sessions
  P_psgn = 50;               % Max. pseudo GN between pruning sessions
  greps = 1e-4;             % Gradient norm stopping criteria
  kills = 2;                % Number of weights to prune away at a time
  min_dim = 7;              % Minimum dimension of network
  
  doplot_train=0;           % Plots the evolution of training
  
  
  randn('seed',sum(100*clock));
  
  % make figures
  if doplot_train~=0
    figh1=figure;
  else
    figh1=0;
  end
  
  
  % First, get some data...
  N1 = 100;
  N2 = 12;
  X_tr = [ randn(N1, 1) - 2 ; randn(N2, 1) + 2];
  y_tr = [ ones(N1, 1) ; 1+ones(N2, 1) ];
  
  meanX = mean(X_tr, 1);
  stdX  = std(X_tr, 1);
  train_inp = X_tr; %
  train_tar = y_tr;
  
  clear Etrain Etest Error_train Error_test

  
  % Initialize network weights
  [Wi,Wo] = nc_winit(Ni,Nh,No,range,seed);

  % Perform initial training of the net
  [Wi,Wo] = nc_train(Wi,Wo,alpha_i,alpha_o,train_inp, train_tar, ...
      I_gr, I_psgn,greps,figh1);
  
  dim = nc_dimen(Wi,Wo);
  iter = 1;
  dimvec(iter) = dim;
  Etrain(iter) = nc_cost_c(Wi,Wo,alpha_i,alpha_o,train_inp,train_tar);
  Error_train(iter) = nc_err_frac(Wi,Wo,train_inp,train_tar);

    

  % Construct grid
  grid_inp = (-5:0.1:5)';
  [Vj, grid_outy] = nc_forward(Wi, Wo, grid_inp);
  grid_out = nc_softmax(grid_outy);

  figure
  subplot(3,1,1)
  plot(grid_inp,  1/sqrt(2*pi)*exp(-0.5 * (grid_inp+2).^2), 'b-')
  hold on
  plot(grid_inp,  1/sqrt(2*pi)*exp(-0.5 * (grid_inp-2).^2), 'r--')
  plot(train_inp(1:N1), 0, 'bx', train_inp(N1+1:end), 0, 'ro')
  xlabel('Input')
  ylabel('Input density p(x)')
  hold off
  axis([-5 5 0 0.5])
  ax = axis;
  
  subplot(3,1,2)
  plot(grid_inp, grid_out(:,1), 'b-x', ...
      grid_inp, grid_out(:,2), 'r--o')
  xlabel('Input')
  ylabel('Network posterior probability')
  axis([ax(1:2) 0 1])

  true_out1=1./(1 + (N2/N1)*exp(-0.5*(grid_inp-2).^2 + 0.5*(grid_inp+2).^2));
  true_out2=ones(size(true_out1))-true_out1;
  subplot(3,1,3);
  plot(grid_inp, true_out1, 'b-x', ...
      grid_inp, true_out2, 'r--o')
  xlabel('Input')
  ylabel('True posterior probability')
  axis([ax(1:2) 0 1])





    



