function [Wi_tr,Wo_tr] = nr_train(Wi, Wo, alpha_i, alpha_o, Inputs, ...
    Targets, gr, psgn, neps)

%NR_TRAIN      Train network
%   [Wi_tr,Wo_tr] = nr_train(Wi, Wo, alpha_i, alpha_o, Inputs, ...
%   Targets, gr, psgn, neps) trains a network with gradient descent
%   followed by pseudo Gauss-Newton
%   
%   Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%        gr      :  Max. number of gradient descent steps
%        psgn    :  Max. number of pseudo Gauss-Newton steps
%        neps    :  Gradient norm stopping criteria 
%   Output:
%        Wi_tr   :  Matrix with trained input-to-hidden weights
%        Wo_tr   :  Matrix with trained hidden-to-outputs weights
%
%   Neural Regression toolbox, DSP IMM DTU

  % Create mask to remove non-active weights
  i_mask = (Wi ~= 0);
  o_mask = (Wo ~= 0);

  %%%%%%%%%%%%%%%%%%%%%%%%
  %%% GRADIENT DESCENT %%%
  %%%%%%%%%%%%%%%%%%%%%%%%

  [dWi,dWo] =  nr_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
  dWi = dWi .* i_mask;    % Remove non-active terms
  dWo = dWo .* o_mask;

  % Initialization
  gradnorm = nr_two_norm(dWi,dWo);  % Length of gradient
  index = 1;
  gcount = 1;

  while (gradnorm > eps) & (gcount <= gr)

    % Save results so far...
    Etrain(index) = nr_cost_e(Wi,Wo,Inputs,Targets);
    normtrace(index)=gradnorm;

    % Make plots every now and then...
    if rem(gcount,4) == 0
      figure(1)
      semilogy(0:index-1,Etrain)
      title('Training error')
      drawnow
    end
    if rem(gcount,10) == 0
      figure(2)
      semilogy(0:index-1,normtrace);
      title('Gradient norm')
      drawnow
    end

    % Determine steplength
    eta = nr_linesear(Wi,Wo,-dWi,-dWo,alpha_i,alpha_o,Inputs,Targets,12);
   
    % Update weights
    Wi = Wi - eta * dWi;
    Wo = Wo - eta * dWo;

    [dWi,dWo] =  nr_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
    dWi = dWi .* i_mask;    % Remove non-active terms
    dWo = dWo .* o_mask;
  
    gradnorm = nr_two_norm(dWi,dWo);  
    gcount = gcount + 1;
    index = index + 1;
  end  

  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% PSEUDO GAUSS-NEWTON %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%

  [dWi,dWo,ddWi,ddWo] =  nr_pseuhess(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
  dWi = dWi .* i_mask;   % Remove non-active terms
  dWo = dWo .* o_mask;

  while (gradnorm > eps) & (gcount <= psgn)

    % Save results so far...
    Etrain(index) = nr_cost_e(Wi,Wo,Inputs,Targets);
    normtrace(index) = gradnorm;

    % Make plots every now and then...
    if rem(gcount,4) == 0
      figure(1)
      semilogy(0:index-1,Etrain)
      title('Training error')
      drawnow
    end
    if rem(gcount,10) == 0
      figure(2)
      semilogy(0:index-1,normtrace);
      title('Gradient norm')
      drawnow
    end

    % Calculate search direction
    Di = -(dWi ./ ddWi);
    Do = -(dWo ./ ddWo);
  
    % Determine steplength
    eta = nr_linesear(Wi,Wo,Di,Do,alpha_i,alpha_o,Inputs,Targets,12);
   
    % Update weights
    Wi = Wi + eta * Di;
    Wo = Wo + eta * Do;
 
    [dWi,dWo,ddWi,ddWo] = nr_pseuhess(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
    dWi = dWi .* i_mask;    % Remove non-active terms
    dWo = dWo .* o_mask;

    gradnorm = nr_two_norm(dWi,dWo);  
    index = index + 1;
    gcount = gcount + 1;
  end

  Wi_tr = Wi;
  Wo_tr = Wo;




