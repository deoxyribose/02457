%SUN_ITERATE   Main for neural network optimization comparison
%
%
%  See also: NR_TRAINX

%  cvs: 
  
  
  %%%%%%%%%%%%%%%%%%%%%
  % Initialization    %
  %%%%%%%%%%%%%%%%%%%%%
  
  Ni = 12;                  % Number of external inputs
  Nh = 3;                   % Number of hidden units
  No = 1;                   % Number of output units (classes -1)
  
  % Weight initialization
  range=.5;
  
  alpha_i = 1e-2;           % Input weight decay
  alpha_o = 1e-2;           % Output weight decay
  
  iterations = 100; 
  greps = 1e-4;             % Gradient norm stopping criteria
  
  %%%%%%%%%%%%%%%%%%%%
  % Load data        %
  %%%%%%%%%%%%%%%%%%%%
  
  load sp
  x0 = x0(1:12,:)';
  x1 = x1(1:12,:)';
  x2 = x2(1:12,:)';
  y0 = y0';
  y1 = y1';
  y2 = y2';
  
  
  %%%%%%%%%%%%%%%%%%%%

  % Training of the net
  N = 3;
  E = zeros(5,N,iterations);
  for n = 1:N
    % Initialize network weights
    seed = sum(100*clock);
    [Wi_init,Wo_init] = nr_winit(Ni, Nh, No, range, seed);
    
    % Gradient
    [Wi, Wo, Etrain] = nr_trainx(Wi_init, Wo_init, alpha_i, alpha_o, ...
	x0, y0, iterations, 0, greps, 1);
    E(1, n, :) = Etrain;
    
    % Pseudo-Gauss-Newton
    [Wi, Wo, Etrain] = nr_trainx(Wi_init, Wo_init, alpha_i, alpha_o, ...
	x0, y0, 10, iterations-10, greps, 1);
    E(2, n, :) = Etrain;

    % Fletcher-Reeves
    [Wi, Wo, Etrain] = nr_trainx(Wi_init, Wo_init, alpha_i, alpha_o, ...
	x0, y0, 0, iterations, greps, 1, 'FR');
    E(3, n, :) = Etrain;

    % Hestenes-Stiefel
    [Wi, Wo, Etrain] = nr_trainx(Wi_init, Wo_init, alpha_i, alpha_o, ...
	x0, y0, 0, iterations, greps, 1, 'HS');
    E(4, n, :) = Etrain;

    % Polak-Ribiere
    [Wi, Wo, Etrain] = nr_trainx(Wi_init, Wo_init, alpha_i, alpha_o, ...
	x0, y0, 0, iterations, greps, 1, 'PR');
    E(5, n, :) = Etrain;
    
  end


  figure
  plot(squeeze(mean(E, 2))')
    plot(1:size(E, 3), squeeze(mean(E(1,:,:), 2))', '+-', ......
      1:size(E, 3), squeeze(mean(E(2,:,:), 2)), 'o-', ......
      1:size(E, 3), squeeze(mean(E(3,:,:), 2)), ':', ...
      1:size(E, 3), squeeze(mean(E(4,:,:), 2)), '--', ...
      1:size(E, 3), squeeze(mean(E(5,:,:), 2)), '-')

  xlabel('Iterations')
  ylabel('Cost')
  legend('Gradient', 'PGN', 'FR', 'HS', 'PR'); 

