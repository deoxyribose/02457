
%MAIN7B        Comparison of entropic and square costfunction.
%
%  04364, DSP IMM DTU, 1999, Finn

% cvs: $Revision: 1.2 $



  clear trace_e
  N = 21;

  trace_tar = ones(N,1);
  trace_tar_sq = [trace_tar trace_tar*0];
  trace_outy = linspace(-10,10,N)';
  trace_out = nc_softmax(trace_outy);
  
  for n = 1:size(trace_out,1)

    % 
    Targets = trace_tar(n,:);
    yj = trace_outy(n,1);
    
    % Code taken from "nc_cost_e"
      % Expand output unit outputs with zeros for the last class
      yj = [yj zeros(max(size(yj)),1)];
      
      [exam,c]=size(yj);
      
      % Offsets for numerical trick
      offsets = max(yj')';
      offsets_expand=kron(ones(1,c),offsets);
      
      % Determine column vector with outputs corresponding to 
      % the correct classes
      for i=1:exam
	correct_out(i)=yj(i,Targets(i));
      end
      correct_out = correct_out'; 
      
      % Calculate individual likelihoods using numerical trick
      likelihoods = offsets + log( sum( exp(yj-offsets_expand)' )' ) - ...
	  correct_out; 
      
      % Calculate mean sum of likelihoods
      error = sum(likelihoods)/exam;
    
    %%%
    
    % Entropic error
    trace_e_en(n) = error;
    
    % Square error
    trace_e_sq(n) = 0.5 * sum(sum((trace_tar_sq(n,:) - trace_out(n,:)) .^ 2));
  end

  subplot(2,1,1)
  plot(trace_outy, trace_e_sq, 'r-x', ...
      trace_outy, trace_e_en, 'b--o')
  xlabel('Neural network output before softmax (\phi_1)')
  ylabel('Cost function value')

  
  subplot(2,1,2)
  plot(trace_out(:,1), trace_e_sq, 'r-x', ...
      trace_out(:,1), trace_e_en, 'b--o')
  xlabel('Output from first unit after softmax = probability (y_1)')
  ylabel('Cost function value')
  













