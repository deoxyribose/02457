
%NC_NETPRUN    Neural classifier main program
%  Main program for neural network training and pruning
%  This program implements classification using the SOFTMAX function
%
%  The network is initially trained using gradient descent and 
%  pseudo Gauss-Newton, using line search to determine step lengths.
%  Afterwards, pruning is performed using the Optimal Brain Damage 
%  method.
%
%  Neural classifier, DSP IMM DSP,  Programmed February 1997 by 
%                                   Morten With Pedersen


  clear
  % Glass data
  Ni = 9;                   % Number of external inputs
  Nh = 4;                   % Number of hidden units
  No = 5;                   % Number of output units
  
  % Weight initialization
  seed=sum(100*clock);
  range=0.3;
  
  alpha_i = 1e-3;           % Input weight decay
  alpha_o = 1e-3;           % Output weight decay
  
  I_gr = 5;                % Initial max. gradient iterations
  I_psgn = 15;             % Initial max. pseudo Gauss-Newt iterations
  P_gr = 0;                 % Max. gradient it. between pruning sessions
  P_psgn = 50;              % Max. pseudo GN between pruning sessions
  greps = 1e-4;             % Gradient norm stopping criteria
  kills = 2;                % Number of weights to prune away at a time
  min_dim = 7;              % Minimum dimension of network
  
  doplot_train=1;           % Plots the evolution of training
  doplot_prune=1;           % Plots the evolution of pruning
  
  
  %%%%%%%% Save the parameters in a .ini file %%%%%%%%%%%%
  save netprun.ini
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
  [train_inp,train_tar,test_inp,test_tar] = nc_getdata;  % Glass data

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
  
  iter = iter + 1;
  
  while dim > min_dim+kills
    % Prune weights away
    [Wi,Wo] = nc_prune(Wi,Wo,alpha_i,alpha_o,train_inp,train_tar,kills);
    
    [Wi,Wo] = nc_train(Wi, Wo, alpha_i, alpha_o, train_inp, ...
	train_tar, P_gr, P_psgn, greps, figh1); 

    dim = nc_dimen(Wi,Wo);     % Determine new dimension
    dimvec(iter) = dim;
    Etrain(iter) = nc_cost_c(Wi,Wo,alpha_i,alpha_o,train_inp,train_tar);
    Etest(iter) = nc_cost_e(Wi,Wo,test_inp,test_tar);
    Error_train(iter) = nc_err_frac(Wi,Wo,train_inp,train_tar);
    Error_test(iter) = nc_err_frac(Wi,Wo,test_inp,test_tar);
    
    if doplot_prune
      figure(figh2)
      subplot(2,1,1)
      semilogy(dimvec,Etrain)
      hold on
      semilogy(dimvec,Etest,':')
      hold off
      title('COST')
      subplot(2,1,2)
      title('ERROR  RATE')
      plot(dimvec,Error_train)
      hold on
      plot(dimvec,Error_test,':')
      hold off  
      title('ERROR  RATE')
      drawnow
    else
      disp([sprintf('%3.0f',iter) ...
	    sprintf(' %1.4f', Etrain(iter)) ...
	    sprintf(' %1.4f', Etest(iter)) ...
	    sprintf(' %1.4f', Error_train(iter)) ...
	    sprintf(' %1.4f',Error_test(iter))])
    end
    
    iter = iter + 1;
  end
