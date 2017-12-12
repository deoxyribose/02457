function nr_plotnet(Wi, Wo, plottype)


%NR_PLOTNET    Plot network
%   NR_PLOTNET(Wi, Wo, plottype)
%
%   Input:  
%        Wi       :  Matrix with input-to-hidden weights
%        Wo       :  Matrix with hidden-to-outputs weights
%        plottype :  type of plot
%
%    See also NR_TRAIN
%
%    Neural Regression toolbox, DSP IMM DTU

% cvs: $Id: nr_plotnet.m,v 1.1 1999/10/19 18:22:36 fnielsen Exp $

  % Default values
  if nargin < 3
    plottype = 1;
  end
  
  Wi = Wi';
  Wo = Wo';
  
  % Find network size
  [Nii, Nh] = size(Wi);
  [Nhh, No] = size(Wo);
  Ni = Nii - 1;
  
  if plottype == 1
    
    clf
    set(gcf, 'color', [1 1 1]);
    
    linethickness = 5;
    linethickness = linethickness / max([ abs(Wi(:)) ; abs(Wo(:))]);
    
    maxunits = max([Nii,Nhh,No]);	
    
    % Coordinates
    XI = 30;
    XH = 70;
    Xo = 110;
    YI = zeros(Nii,1);
    YH = zeros(Nhh,1);
    Yo = zeros(No,1);	

    text(35, -10, 'Input weights');
    text(80, -10, 'Output weights');
    
    % Draw units
    for i = 1 : Nii
      YI(i) = 100 * (i-0.5)/Nii;
      lyngby_circle(XI, YI(i), 1);
    end
    for j = 1 : Nhh
      YH(j) = 100 * (j-0.5)/Nhh;
      lyngby_circle(XH, YH(j), 1);
    end
    for k = 1 : No
      Yo(k) = 100 * (k-0.5)/No;
      lyngby_circle(Xo, Yo(k), 1);
    end
    YI = flipud(YI);
    YH = flipud(YH);
    Yo = flipud(Yo);
    
    if Nhh>Nh
      text(65, YH(Nhh)-3, 'bias');
    end
    if Nii>Ni
      text(25, YI(Nii)-3, 'bias');
    end
    
    % Draw Weights
    for i = 1 : Nii
      for j = 1 : Nh
	if Wi(i,j) > 0
	  line([XI XH],[YI(i) YH(j)], 'Color', [1 0 0], ...
	      'LineStyle', '-', ...
	      'LineWidth', linethickness*abs(Wi(i,j)));
	elseif Wi(i,j) < 0
	  line([XI XH],[YI(i) YH(j)], 'Color', [0 0 1], ...
	      'LineStyle', '--', ...
	      'LineWidth', linethickness*abs(Wi(i,j)));
	end			
      end
    end
    for h = 1 : Nhh
      for o = 1 : No
	if Wo(h,o) > 0
	  line([XH Xo],[YH(h) Yo(o)], 'Color', [1 0 0], ...
	      'LineStyle', '-', ...
	      'LineWidth', linethickness*abs(Wo(h,o)));
	elseif Wo(h,o) < 0
	  line([XH Xo],[YH(h) Yo(o)],'Color', [ 0 0 1], ...
	      'LineStyle', '--', ...
	      'LineWidth', linethickness*abs(Wo(h,o)));
	end			
      end
    end

    hPos = line([0 0],[0 0], 'Color', [1 0 0], ...
	'LineStyle', '-', ...
	'LineWidth', 2, 'HandleVisibility', 'off');
    hNeg = line([0 0],[0 0], 'Color', [0 0 1], ...
	'LineStyle', '--', ...
	'LineWidth', 2, 'HandleVisibility', 'off');

    
    legend([hPos hNeg], 'Positive', 'Negative');
    axis off;		
    axis([28 112 -5 105]);

    
  elseif plottype == 2

    % Colormap ?
    h = hot(36);
    h = h(1:32,:);
    c = [flipud(fliplr(h)) ; h];
    
    maxWi = max(abs(Wi(:)));
    maxWo = max(abs(Wo(:)));
    subplot(1,2,1) 
    imagesc([Wi zeros(Nii, Nhh-Nh)]', [-maxWi maxWi] )
    axis([0.5 Nii+0.5 0.5 Nhh+0.5]),
    title('Input weights')
    xlabel('Input index')
    ylabel('Hidden index')
    colorbar
    subplot(1,2,2)
    imagesc(Wo, [-maxWo maxWo])
    title('Output weights')
    xlabel('Output index')
    ylabel('Hidden index')
    colorbar
    colormap(c)
    
  end
  






