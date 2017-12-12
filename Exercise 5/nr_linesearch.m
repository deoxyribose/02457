function [f2,df2,WW,eta] = nr_linesearch(WW, D, alpha_i, alpha_o, ...
    Inputs, Targets, h0, slopeX, eta, fX, GX, WWmask)  

%NR_LINESEARCH Line search with Wolfe-Powell conditions
%   [f2,df2,WW,eta] = nr_linesearch(WW, D, alpha_i, alpha_o, Inputs,
%       Targets, h0, slopeX, eta, fX, GX, WWmask)
%
%   Input:
%     WW     :  weight vector (the matrices matrices reshaped into one vector)
%     D      :  the vector with stored dimensions of the weight matreces
%     h0     :  direction vector
%     slopeX :
%     eta    :  guessed step size
%     fX     :  function value for the starting point
%     GX     :  gradient at the starting point
%     WWmask :  weights mask
%
%   Output :
%     f2     : function value for the minimizer
%     df2    : gradient for the minimizer
%     WW     : output weights 
%     eta    : output step size

%   cvs: $Revision: 1.1 $

  MAXEVAL = 20 ;            % Max 20 function evaluations in 1 line search
  SIGMA = 0.5 ;             % Target attenuation of gradient. (def. .5)
  RHO = 0.25 ;              % Max decrease from current slope. (def. .25)
  INTLIM = 0.1 ;            % Interpolation limit 0.1=10% interval length.
  EXTLIM = 3.0 ;            % Extrapolation limit 3.0=3 interval length.
  LSSUCCESS = 0 ;           % Success flag (Line search).
  nbeval = 1;               % One already have been done before entering the function
  
  [f2,df2] = nr_calcul(WW,D,alpha_i, alpha_o,Inputs,Targets);
  nbeval = nbeval + 1 ;
  
  slope2 = df2 * h0' ;            % Slope at current guess.
  f3 = fX ; slope3 = slopeX ; eta3 = -eta ;       % Set point 3.
  
  maxeta = -1 ;
  
  while ((nbeval <= MAXEVAL) & ~LSSUCCESS)
    while (((f2 > fX + eta * RHO * slopeX) | ...   % Condition b(X).
            (slope2 > -SIGMA * slopeX)) & ...       % Condition a2(X).
            (nbeval <= MAXEVAL))
         
      maxeta = eta ;
      
      if (f2 > fX)                % Quadratic interpolation.
        neweta = eta3 - (slope3 * eta3*eta3 / 2) / ...
            (slope3 * eta3 - f3 + f2) ;
      else                        % Cubic interpolation.
        a = 6*(f2 - f3) / eta3 + 3*(slope2 + slope3) ;
        b = 3*(f3 - f2) - eta3 * (2*slope2 + slope3) ;
        neweta = (sqrt(b*b - a * slope2 * eta3*eta3) - b) / a ;
      end
      
      % We will now place the new current point (indexed 2).
      if ~isreal(neweta)                       % Numerical error.
        neweta = eta3 / 2 ;
      elseif (neweta > INTLIM * eta3)       % too close to current.
        neweta = INTLIM * eta3 ;
      elseif (neweta < (1-INTLIM) * eta3)   % too close to x3.
        neweta = (1-INTLIM) * eta3 ;
      else
        neweta = neweta ;
      end
      
      eta = eta + neweta ;     % Update global step value.
      WW = WW + neweta * h0 ;    % Take another step, compute, etc.
      WW = WW.*WWmask;

      [f2,df2] = nr_calcul(WW,D,alpha_i, alpha_o,Inputs,Targets);
      nbeval = nbeval + 1 ;
      
      slope2 = df2 * h0' ;       
      eta3 = eta3 - neweta ;           % Narrow the interpolation.
    end
   
   
    if (slope2 > SIGMA * slopeX)              % Condition a1(X)
      LSSUCCESS = 1 ;             % Let's get outta here.
    else                          % Cubic extrapolation.
      a = 6*(f2 - f3) / eta3 + 3*(slope2 + slope3) ;
      b = 3*(f3 - f2) - eta3 * (2*slope2 + slope3) ;
      neweta = -slope2 * eta3*eta3 / ...
         (sqrt(b*b - a * slope2 * eta3*eta3) + b) ;
      
      
      if ~isreal(neweta)                       % Numerical error.
        if (maxeta < 0)                          % No limit set ?
          neweta = eta * (EXTLIM-1) ;
        else
          neweta = (maxeta - eta) / 2 ;
        end
      elseif (neweta < 0)                    % Wrong side !
        if (maxeta < 0)
          neweta = eta * (EXTLIM-1) ;
        else
          neweta = (maxeta - eta) / 2 ;
        end
      elseif ((maxeta >= 0) & (eta + neweta) > maxeta)
        neweta = (maxeta - eta) / 2 ;      % If extrap. beyond max step.
      elseif ((maxeta < 0) & (eta + neweta) > eta*EXTLIM)
        neweta = eta * (EXTLIM - 1) ;
      elseif (neweta < -eta3 * INTLIM)      % Too close from current.
        neweta = -eta3 * INTLIM ;
      elseif ((maxeta >= 0) & (neweta < (maxeta - eta) * (1-INTLIM)))
        neweta = (maxeta - eta) * (1-INTLIM) ; % Too close to max.
     end
     
      f3 = f2 ; slope3 = slope2 ;             % Point 3 <- Point 2.
      eta3 = -neweta ;                      % Relative to current.
      eta = eta + neweta ;     % Update global step value.
      
      WW = WW + neweta * h0 ;    % Take another step, compute, etc.
      WW = WW.*WWmask;
      
      [f2,df2] = nr_calcul(WW,D,alpha_i, alpha_o,Inputs,Targets);
      nbeval = nbeval + 1 ;
      
      slope2 = df2 * h0' ;      
    end
 end








