function [Wi_tr,Wo_tr, Etrain] = nr_trainx(Wi, Wo, alpha_i, alpha_o, ...
    Inputs, Targets, gr, psgn, neps, figh, method)

%NR_TRAINX     Train network (conjugate gradient version)
%   [Wi_tr,Wo_tr] = nr_trainx(Wi, Wo, alpha_i, alpha_o, Inputs,
%   Targets, gr, psgn, neps, figh)
%   Train the network with gradient descent followed by pseudo
%   Gauss-Newton 
%
%   Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%        psgn    :  Max. number of steps in pseudo Gauss-Newton 
%                   or N dimensional passes Conjugate Gardient 
%        gr      :  Max. number of gradient descent steps
%                   not used for CG 
%        neps    :  Gradient norm stopping criteria not used for CG
%        figh    :  figure handle. If no plotting is desired use 
%                   figh=0;
%        method  :  defined only when Conjugative Gradient method used
%                       FR - Fletcher-Reeves
%		        HS - Hestenes-Stiefel 
%			PR - Polak-Ribiere (default)
%                   
%    Output:
%        Wi_tr   :  Matrix with trained input-to-hidden weights
%        Wo_tr   :  Matrix with trained hidden-to-outputs weights
%      
%    Neural Regression toolbox, DSP IMM DTU               

%    JL97, MWP97, Anna 1999
         
%    cvs: $Revision: 1.3 $

  [Ni,Mi] = size(Wi);
  [No,Mo] = size(Wo);
  D = [Ni,Mi,No,Mo];

  tol = 1e-16;

  Etrain = [];
  
  if ~exist('figh')
    figh=0;
  end

  % Create mask to remove non-active weights
  i_mask = (Wi ~= 0);
  o_mask = (Wo ~= 0);

  [dWi,dWo] =  nr_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
  dWi = dWi .* i_mask;    % Remove non-active terms
  dWo = dWo .* o_mask;

  % Initialization
  index = 1;

  if exist('method')
    disp('Performing Conjugate Gradient')

    g0 = - [ dWi(:)'  dWo(:)' ];
    fX = nr_cost_c(Wi, Wo, alpha_i, alpha_o, Inputs, Targets);
    h0 = g0; 
    slopeX = -g0*h0';
    eta = 1/(1-slopeX);
    
    WW = [ Wi(:)' Wo(:)' ];
    WWmask = [ i_mask(:)' o_mask(:)'];

    flag=0;

    while (index <= psgn) & (flag ~= 1)

      [Wi,Wo] = nr_extract(WW,D);

      % Plot and Save results so far...
      if figh~=0
	Etrain(index) = nr_cost_c(Wi, Wo, alpha_i, alpha_o, Inputs, Targets);
	normtrace(index) = sum(g0.^2);  % Norm of gradient

	% Make plots every now and then...
	if rem(index,10) == 0
	  figure(figh);

	  % Plot Cost function
	  subplot(2,1,1)
	  semilogy(0:index-1,Etrain)
	  ylabel('Train Cost')
	  drawnow

	  % Plot gradient norm
	  subplot(2,1,2)
	  semilogy(0:index-1,normtrace);
	  ylabel('Grad. Norm')
	  drawnow
	end
      end
      
      
      WW = WW + eta * h0 ;            % update
      WW = WW.*WWmask;

      [fW,df2,WW,eta] = nr_linesearch(WW, D, alpha_i, alpha_o, Inputs, ...
	  Targets, h0, slopeX, eta, fX, g0, WWmask);

      g1 = -df2; 

      
      % more steps needed?
      if ((abs(fX - fW) < tol*(abs(fW)+abs(fX)+3e-16)) | (g1*g1' == 0)  )
	flag=1; % end of while loop 
	
      else %proceed futher
	
	if method == 'FR' 
	  gamma = (g1*g1')/(g0*g0' + eps);            % Fletcher-Reeves

	elseif method == 'HS'
	  gamma = (g1*(g1-g0)')/(h0*(g1-g0)' + eps);  % Hestenes-Stiefel
	else 
	  gamma = (g1*(g1-g0)') / (g0*g0' + eps);     % Polak & Ribiere statement 
	end
	
	% Updating direction vector, cost function value and gradient vector
	h0 = gamma * h0 + g1; 
	g0 = g1;  
	slope2 = -g0*h0';
	
	if (slope2 > 0)             % Update if not decreasing
	  h0 =  g0 ;
	  slope2 = - g0 * h0' ;
	end

	% To avoid errors when gradient is close to zero
	eta = min(slopeX/slope2,100)*eta; 
	slopeX = slope2;
	fX = fW;    
      end

      index = index + 1;
      
    end
    [Wi,Wo] = nr_extract(WW,D);
    disp(sprintf('Have done %d Conjugate Gradient iterations',index-1))
  else
    if gr~=0
      disp('Performing gradient descent')
    end
    
    gradnorm = nr_two_norm(dWi,dWo);  % Length of gradient
    gcount = 1;

    while (gradnorm > neps) & (gcount <= gr)

      % Save results so far...
      if figh~=0
	Etrain(index) = nr_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
	normtrace(index) = gradnorm;

	% Make plots every now and then...
	if rem(gcount,10) == 0
	  figure(figh);
	  subplot(2,1,1)
	  semilogy(0:index-1,Etrain)
	  ylabel('Train Cost')
	  drawnow

	  subplot(2,1,2)
	  semilogy(0:index-1,normtrace);
	  ylabel('Grad. Norm')
	  drawnow
	end
      end

      % Determine steplength
      eta = nr_linesear(Wi,Wo,-dWi,-dWo,alpha_i,alpha_o,Inputs,Targets,12);
      
      % Update weights
      Wi = Wi - eta * dWi;
      Wo = Wo - eta * dWo;

      [dWi,dWo] = nr_gradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
      dWi = dWi .* i_mask;    % Remove non-active terms
      dWo = dWo .* o_mask;
      
      gradnorm = nr_two_norm(dWi,dWo);  
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

      [dWi,dWo,ddWi,ddWo] =  nr_pseuhess(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
      dWi = dWi .* i_mask;   % Remove non-active terms
      dWo = dWo .* o_mask;

      gcount=1;
      while (gradnorm > neps) & (gcount <= psgn)

	% Save results so far...
	if figh~=0
	  Etrain(index) = nr_cost_c(Wi,Wo,alpha_i,alpha_o,Inputs,Targets);
	  % Eclass(index) = cl_error(Wi,Wo,Inputs,Targets);
	  normtrace(index) = gradnorm;

	  % Make plots every now and then...
	  if rem(gcount,10) == 0
	    figure(figh);
	    subplot(2,1,1)
	    semilogy(0:index-1,Etrain)
	    ylabel('Train Cost')
	    drawnow

	    subplot(2,1,2)
	    semilogy(0:index-1,normtrace);
	    ylabel('Grad. Norm')
	    drawnow
	  end
	end

	% Calculate search direction
	Di = -(dWi ./ ddWi);
	Do = -(dWo ./ ddWo);
	
	% Determine steplength
	eta = nr_linesear(Wi,Wo,Di,Do,alpha_i,alpha_o,Inputs,Targets,12);
	
	% Update weights
	Wi = Wi + eta * Di;
	Wo = Wo + eta * Do;
	
	[dWi,dWo,ddWi,ddWo] = nr_pseuhess(Wi, Wo, alpha_i, alpha_o, ...
	    Inputs, Targets);
	dWi = dWi .* i_mask;    % Remove non-active terms
	dWo = dWo .* o_mask;

	gradnorm = nr_two_norm(dWi,dWo);  
	index = index + 1;
	gcount = gcount + 1;
	
      end
    
      if gcount-1~=0
	disp(sprintf('Have done %d pseudo GN iterations',gcount-1))
      end


    end

  end


  Wi_tr = Wi;
  Wo_tr = Wo;











