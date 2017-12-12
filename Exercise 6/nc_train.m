function [Wi_tr,Wo_tr] = nc_train(Wi, Wo, alpha_i, alpha_o, Inputs, ...
    Targets, gr, psgn, neps, figh) 

%NC_TRAIN      Neural classifier training
%  [Wi_tr,Wo_tr] = train(Wi, Wo, alpha_i, alpha_o, Inputs, Targets,
%    gr, psgn, neps, figh)
%  Train the network with gradient descent followed by pseudo Gauss-Newton
%
%  Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%        gr      :  Max. number of gradient descent steps
%        psgn    :  Max. number of pseudo Gauss-Newton steps
%        neps    :  Gradient norm stopping criteria 
%        figh    :  figure handle. If no plotting is desired use 
%                   figh=0;
%                   
%  Output:
%        Wi_tr   :  Matrix with trained input-to-hidden weights
%        Wo_tr   :  Matrix with trained hidden-to-outputs weights
%                                        
%  Neural Classifier, DSP IMM DTU, JL97, MWP97

%  cvs: $Revision: 1.1 $

if ~exist('figh')
  figh=0;
end

  % Create mask to remove non-active weights
  i_mask = (Wi ~= 0);
  o_mask = (Wo ~= 0);

  %%%%%%%%%%%%%%%%%%%%%%%%
  %%% GRADIENT DESCENT %%%
  %%%%%%%%%%%%%%%%%%%%%%%%
  if gr~=0
    disp('Performing gradient descent')
  end
  [dWi,dWo] =  nc_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
  dWi = dWi .* i_mask;    % Remove non-active terms
  dWo = dWo .* o_mask;

  % Initialization
  gradnorm = nc_eucnorm(dWi,dWo);  % Length of gradient
  index = 1;
  gcount = 1;

  while (gradnorm > neps) & (gcount <= gr)

    % Save results so far...
    if figh~=0
      Etrain(index) = nc_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
      Eclass(index) = nc_cl_error(Wi,Wo,Inputs,Targets);
      normtrace(index)=gradnorm;

      % Make plots every now and then...
    if rem(gcount,10) == 0
      figure(figh);
      subplot(3,1,1)
      semilogy(0:index-1,Etrain)
      ylabel('Train Cost')
      drawnow
      subplot(3,1,2)
      plot(0:index-1,Eclass);
      ylabel('#Misclass')
      drawnow
      subplot(3,1,3)
      semilogy(0:index-1,normtrace);
      ylabel('Grad. Norm')
      drawnow
    end
   end

    % Determine steplength
    eta = nc_linesear(Wi,Wo,-dWi,-dWo,alpha_i,alpha_o,Inputs,Targets,12);
   
    % Update weights
    Wi = Wi - eta * dWi;
    Wo = Wo - eta * dWo;

    [dWi,dWo] =  nc_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
    dWi = dWi .* i_mask;    % Remove non-active terms
    dWo = dWo .* o_mask;
  
    gradnorm = nc_eucnorm(dWi,dWo);  
    gcount = gcount + 1;
    index = index + 1;
  end  
  if (gcount-1)~=0
    disp(sprintf('Have done %d gradient iterations',gcount-1))
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% PSEUDO GAUSS-NEWTON %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  if psgn~=0
    disp('Performing pseudo Gauss-Newton')
  end

  [dWi,dWo,ddWi,ddWo] =  nc_pseuhess(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
  dWi = dWi .* i_mask;   % Remove non-active terms
  dWo = dWo .* o_mask;

  gcount=1;
  while (gradnorm > neps) & (gcount <= psgn)

    % Save results so far...
    if figh~=0
      Etrain(index) = nc_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
      Eclass(index) = nc_cl_error(Wi,Wo,Inputs,Targets);
      normtrace(index)=gradnorm;

      % Make plots every now and then...
    if rem(gcount,10) == 0
      figure(figh);
      subplot(3,1,1)
      semilogy(0:index-1,Etrain)
      ylabel('Train Cost')
      drawnow
      subplot(3,1,2)
      plot(0:index-1,Eclass);
      ylabel('#Misclass')
      drawnow
      subplot(3,1,3)
      semilogy(0:index-1,normtrace);
      ylabel('Grad. Norm')
      drawnow
    end
   end

   % Calculate search direction
    Di = -(dWi ./ ddWi);
    Do = -(dWo ./ ddWo);
 
    % Determine steplength
    eta = nc_linesear(Wi,Wo,Di,Do,alpha_i,alpha_o,Inputs,Targets,12);
   
    % Update weights
    Wi = Wi + eta * Di;
    Wo = Wo + eta * Do;
 
    [dWi,dWo,ddWi,ddWo] = nc_pseuhess(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
    dWi = dWi .* i_mask;    % Remove non-active terms
    dWo = dWo .* o_mask;

    gradnorm = nc_eucnorm(dWi,dWo);  
    index = index + 1;
    gcount = gcount + 1;
  end
  if gcount-1~=0
    disp(sprintf('Have done %d pseudo GN iterations',gcount-1))
  end


    if figh~=0
      Etrain(index) = nc_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
      Eclass(index) = nc_cl_error(Wi,Wo,Inputs,Targets);
      normtrace(index)=gradnorm;

      % Make plots every now and then...
    if rem(gcount,10) == 0
      figure(figh);
      subplot(3,1,1)
      semilogy(0:index-1,Etrain)
      ylabel('Train Cost')
      drawnow
      subplot(3,1,2)
      plot(0:index-1,Eclass);
      ylabel('#Misclass')
      drawnow
      subplot(3,1,3)
      semilogy(0:index-1,normtrace);
      ylabel('Grad. Norm')
      drawnow
    end
   end

  Wi_tr = Wi;
  Wo_tr = Wo;

