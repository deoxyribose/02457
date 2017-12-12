
%MAIN7D        Neural classifier main program
%
%  DSP IMM DTU, Finn

%  cvs: $Revision: 1.1 $


  clear 
  close all

  Ni = 7;                   % Number of external inputs
  Nh = 2;                   % Number of hidden units
  No = 1;                   % Number of output units
  
  % Weight initialization
  seed=sum(100*clock);
  range=0.3;
  
  alpha_i = 1e-1;           % Input weight decay
  alpha_o = 1e-1;           % Output weight decay
  
  I_gr = 5;                 % Initial max. gradient iterations
  I_psgn = 100;             % Initial max. pseudo Gauss-Newt iterations
  P_gr = 5;                 % Max. gradient it. between pruning sessions
  P_psgn = 50;              % Max. pseudo GN between pruning sessions
  greps = 1e-4;             % Gradient norm stopping criteria
  kills = 2;                % Number of weights to prune away at a time
  min_dim = 3;              % Minimum dimension of network
  
  doplot_train=1;           % Plots the evolution of training
  doplot_prune=1;           % Plots the evolution of pruning
  
  
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

  
    
  if 1
    iter = iter + 1;
  
    while dim > min_dim+kills

      % Calculate second derivatives WITHOUT weight decay term
      [dWi,dWo,ddWi,ddWo] =  nc_pseuhess(Wi,Wo,0,0,train_inp, train_tar);
      
      % Calculate saliencies for the input weights INCLUDING gradient term
      Sal_input = (alpha_i + 0.5 * ddWi .^ 2) .* (Wi .^ 2) - (dWi .* Wi); 
      
      % Calculate saliencies for the output weights INCLUDING gradient term
      Sal_output = (alpha_o + 0.5 * ddWo .^ 2) .* (Wo .^ 2) - (dWo .* Wo);
      
      figure(3)
      nc_plotnet(Sal_input,Sal_output)
      title('Saliency')
      drawnow
      
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

      if Etest(iter) < min(Etest(1:(iter-1)))
	Wi_best = Wi;
	Wo_best = Wo;
      end
      
      if doplot_prune
	figure(figh2)
	subplot(2,1,1)
	semilogy(dimvec, Etrain)
	hold on
	semilogy(dimvec, Etest,':')
	hold off
	title('COST')
	subplot(2,1,2)
	title('ERROR  RATE')
	xlabel('Number of parameters in net')
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
    
      figure(4)
      nc_plotnet(Wi, Wo);
      title('Weights')
      drawnow
    
    end
    
    
  end
  


  figure(5)
  nc_plotnet(Wi_best,Wo_best)  
  title('Weights of best net')

  
  % Calculate second derivatives WITHOUT weight decay term
  [dWi,dWo,ddWi,ddWo] =  nc_pseuhess(Wi_best, Wo_best, 0, 0, ...
      train_inp, train_tar); 
  
  % Calculate saliencies for the input weights INCLUDING gradient term
  Sal_input = (alpha_i + 0.5 * ddWi .^ 2) .* (Wi_best .^ 2) - (dWi .* ...
      Wi_best);  
  
  % Calculate saliencies for the output weights INCLUDING gradient term
  Sal_output = (alpha_o + 0.5 * ddWo .^ 2) .* (Wo_best .^ 2) - (dWo ...
      .* Wo_best); 
  
  figure(6)
  nc_plotnet(Sal_input,Sal_output)
  title('Saliency of best net')
  drawnow
  





