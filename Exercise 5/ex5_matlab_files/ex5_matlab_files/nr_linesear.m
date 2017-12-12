function [eta] = nr_linesear(Wi, Wo, Di, Do, alpha_i, alpha_o, Inputs, ...
    Targets, pat)
%NR_LINESEAR   Simple linesearch
%    [eta] = linesear(Wi,Wo,Di,Do,alpha_i,alpha_o,Inputs,Targets,pat)
%    performs a simple linesearch in a direction in parameter space,
%    determining the 'optimal' steplength by iterative decrease.
%
%    Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        Di      :  Matrix with input search direction
%        Do      :  Matrix with output search direction
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
%        pat     :  Patience; max number of decreases
%    Output:
%        eta     :  'Optimal' step length
%
%    See also NR_TRAIN, NR_COST_C
%
%    Neural Regression toolbox, DSP IMM DTU


  % Determine initial cost function value
  old_cost = nr_cost_c(Wi, Wo, alpha_i, alpha_o, Inputs, Targets);
  
  cost = old_cost+1;   % Make sure first iteration is performed
  eta = 2;             % 2* Initial step length
  k=0;                 % Keep track of number of decreases 
  
  % Decrease eta until cost is diminished
  while (cost > old_cost) & (k <= pat)
    eta = eta / 2;
    k = k + 1;
    cost = nr_cost_c(Wi+eta*Di, Wo+eta*Do, alpha_i, alpha_o, Inputs, ...
	Targets); 
  end

  % Will smaller eta decrease cost further?
  while (cost <= old_cost) & (k <= pat)
    old_cost = cost;
    cost = nr_cost_c(Wi+0.5*eta*Di, Wo+0.5*eta*Do, alpha_i, alpha_o, ...
	Inputs, Targets);
    if (cost <= old_cost)
      eta = 0.5*eta;
    end;
    k = k+1;
  end;

  if k > pat
    eta = 0;     % Ran out of patience...
  end














