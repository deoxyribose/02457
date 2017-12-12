
%MAIN7A        Neural classifier main program
%
%  DSP IMM DTU, Finn

%  cvs: $Revision: 1.1 $

  clear
  close all

  Ni = 7;                   % Number of external inputs
  Nh = 4;                   % Number of hidden units
  No = 1;                   % Number of output units
  
  % Weight initialization
  seed=sum(100*clock);
  range=0.3;
  
  alpha_i = 1e-1;           % Input weight decay
  alpha_i = alpha_i*100;           % Input weight decay
  alpha_o = 1e-1;           % Output weight decay
  alpha_o = alpha_o*100;
  
  I_gr = 5;                 % Initial max. gradient iterations
  I_psgn = 30;              % Initial max. pseudo Gauss-Newt iterations
  P_gr = 0;                 % Max. gradient it. between pruning sessions
  P_psgn = 50;               % Max. pseudo GN between pruning sessions
  greps = 1e-4;             % Gradient norm stopping criteria
  kills = 2;                % Number of weights to prune away at a time
  min_dim = 7;              % Minimum dimension of network
  
  doplot_train=1;           % Plots the evolution of training
  doplot_prune=0;           % Plots the evolution of pruning
  
  
  %%%%%%%% Save the parameters in a .ini file %%%%%%%%%%%%
  % save netprun.ini
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  randn('seed',sum(100*clock));
  
  % make figures
  if doplot_train~=0
    figh1=figure;
  else
    figh1=0;
  end
  if doplot_prune~=0
    figh2=figure;
  end
  
  
  % First, get some data...
  load pima
  meanX = mean(X_tr);
  stdX  = std(X_tr);
  train_inp = (X_tr - repmat(meanX, size(X_tr, 1), 1)) ...
      ./ repmat(stdX, size(X_tr, 1), 1);
  train_tar = y_tr;
  test_inp  = (X_te - repmat(meanX, size(X_te, 1), 1)) ...
      ./ repmat(stdX, size(X_te, 1), 1);
  test_tar  = y_te;

  
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
  Etest(iter) = nc_cost_e(Wi,Wo,test_inp,test_tar);
  
  Error_train(iter) = nc_err_frac(Wi,Wo,train_inp,train_tar);
  Error_test(iter) = nc_err_frac(Wi,Wo,test_inp,test_tar);
  
  if doplot_prune==0
    disp(['iter, ' 'Etrain, ' 'Etest, ' 'Misclas_train, ' 'Misclas_test'])
    disp([sprintf('%3.0f',iter) sprintf(' %1.4f',Etrain(iter)) ...
	  sprintf(' %1.4f', Etest(iter)) ...
	  sprintf(' %1.4f', Error_train(iter)) ...
	  sprintf(' %1.4f', Error_test(iter))])   
  end

  [U, S, V] = svd(test_inp, 0);
  index1 = find(test_tar==1);
  index2 = find(test_tar==2);
  
  figure
  clf
  axis([-0.3 0.3 -0.3 0.3])
  ax = axis;
  M = 3;
  for m = 1:M-1
    for n = (m+1):M 
      plotno = (n-m) + (M-1)*(m-1);
      subplot(M-1,M-1, plotno);
      plot(U(index1,m), U(index1,n), 'bx', ...
	  U(index2,m), U(index2,n), 'go')
      xlabel(sprintf('%d. principal component', m));
      ylabel(sprintf('%d. principal component', n))
      axis(ax)
    end
  end

  % Construct grid
  x = -0.3:0.025:0.3;
  y = -0.3:0.025:0.3;
  [X_grid,Y_grid] = meshgrid(x, y);
  clear grid_inp
  for m = 1:length(x)
    for n = 1:length(y)
      grid_inp(m + (n-1)*length(x), :) = ...
	  x(m) * S(1,1) * V(:,1)' + ...
	  y(n) * S(2,2) * V(:,2)';
    end

  end
  
  [Vj, grid_outy] = nc_forward(Wi, Wo, grid_inp);
  grid_out = nc_softmax(grid_outy);

  figure
  plot(U(index1,1), U(index1,2), 'bx', ...
      U(index2,1), U(index2,2), 'go')
  hold on
  contour(X_grid, Y_grid, reshape(grid_out(:,1), length(x), ...
      length(y))')
  % Plot decision boundary=0
  [c, h] = contour(X_grid, Y_grid, reshape(0.5*(grid_out(:,1)>0.5), ...
      length(x), length(y))',1);
  set(h, 'linewidth', 3);
  hold off
  xlabel('1. principal component')
  ylabel('2. principal component')
  title('Decision boundary')
  axis(ax)

  figure
  subplot(1,2,1)
  surfc(X_grid, Y_grid, reshape(grid_out(:,1), length(x), ...
      length(y))')
  xlabel('1. principal component')
  ylabel('2. principal component')
  title('Class-conditional probability, class 1')
  axis(ax)
  axis([ax 0 1])

  subplot(1,2,2)
  surfc(X_grid, Y_grid, reshape(grid_out(:,2), length(x), ...
      length(y))')
  xlabel('1. principal component')
  ylabel('2. principal component')
  title('Class-conditional probability, class 2')
  axis(ax)
  axis([ax 0 1])