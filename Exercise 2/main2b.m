% MATLAB program for exercise 2 in course 02457
% This program is for part 2 out of 4 
%
% "main2b" visualizes the interpretation of the covariance 
% matrix, SIGMA, of a 2D Normal distribution.
%
% This program allows you to vary the covariance matrix of a
% 2D Normal distribution and makes contour plots of the PDF
% and the histogram of the sample.

% (c) Karam Sidaros, September 1999.
% Uses randmvn.m mvnpdf.m hist2d.m arrow.m clevels.m
%

%%%%%%%%%%%%%%%%%%%%%%%%% Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Interpretation of Covariance %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu=[5 20]';	   % true mean value
SIGMA=[2  2.5  
       2.5 5  ];   % true covariance matrix
N=10000;           % number of points in density

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  disp('The true mean is');
  mu
  disp('and the true covariance matrix is');
  SIGMA
  disp('The number of samples is');
  N
  
  D = randmvn(mu,SIGMA,N);  % A random sample - set of N samples
  
  mu_ = mean(D,2);  % Estimated mean
  SIGMA_ = cov(D'); % Estimated covariance matrix

  [U lambda] = eig(SIGMA_);    % Eigenvectors and eigenvalues

  disp('The estimated mean is');
  mu_
  disp('and the estimated covariance matrix is');
  SIGMA_
  
  fprintf('Calculating histogram ... ');
  nbins = 20;                   % # bins in histogram in each dimension
  [n,x] = hist2d(D,nbins,[],1);     % Normalized histogram
  fprintf('Done\n\r');

  resol = 100;                   % # points in plot
  range = x(:,nbins)-x(:,1);
  x1 = x(1,1):range(1)/(resol-1):x(1,nbins);
  x2 = x(2,1):range(2)/(resol-1):x(2,nbins);
  
  [X1 X2] = meshgrid(x1,x2);
  X1=X1(:);X2=X2(:);

  fprintf('Calculating true PDF ... ');
  p = mvnpdf(mu,SIGMA,[X1';X2']);% True PDF 
  p = reshape(p,[resol resol]);
  fprintf('Done\n\r');
 
  n1 = sum(n,1);
  n2 = sum(n,2)';  
  n1 = n1/sum(n1)/(range(1)/(nbins-1));
  n2 = n2/sum(n2)/(range(2)/(nbins-1));
  
  p1 = mvnpdf(mu(1),std(D(1,:))^2,x1);  % Marginal PDF  
  p2 = mvnpdf(mu(2),std(D(2,:))^2,x2);  
  
  
%%%%%%%%%%%%%%%%%%  Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%

  figure(1)
  clf
  h=get(gcf,'Position');
  h0 = get(0,'defaultfigureposition');
  h(3:4)=[1.5 1].*h0(3:4);
  set(gcf,'Position',h);
  set(gcf,'PaperUnits','centimeters');
  set(gcf,'PaperType','a4');
  set(gcf,'PaperOrientation','landscape');
  set(gcf,'PaperPosition',[1 3 28 14]);
  subplot 231
    plot(D(1,:),D(2,:),'.'); 
    ax=axis;
    axis equal;
    axis(ax);
    h = get(gca,'position');
    h1 = get(gca,'plotboxaspectratio');   
    xlabel('x_1')
    ylabel('x_2')  
    title('Scatter plot of data-set');
  subplot 234
    bar(x(1,:),n1,1,'w');
    pos = get(gca,'position');
    pos([1 3]) = h([1 3]);
    set(gca,'position',pos);    
    set(gca,'plotboxaspectratio',h1);
    xlim(ax(1:2));
    hold on
    h2 = plot(x1,p1,'b');    
    set(h2,'linewidth',2);   
    ylim([0 max([n1 p1])*1.02]);    
    xlabel('x_1')
    ylabel('p(x_1)')
  subplot 232
    barh(x(2,:),n2,1,'w');
    pos = get(gca,'position');
    pos([2 4]) = h([2 4]);
    set(gca,'position',pos);    
    set(gca,'plotboxaspectratio',h1);
    ylim(ax(3:4));
    hold on
    h2 = plot(p2,x2,'b');    
    set(h2,'linewidth',2);    
    xlim([0 max([n2 p2])*1.02]);    
    ylabel('x_2')  
    xlabel('p(x_2)')
  subplot 235
    h2 = text(0.2, 0.8,['\mu = (', num2str(mu'), ')^T']);
    h2(2) = text(0.2,0.6,['\Sigma = ']);
    xtnt = get(h2(2),'extent');
    h2(3) = text(xtnt(1)+1.1*xtnt(3),xtnt(2)+1.0*xtnt(4),num2str(SIGMA(1,:)));
    h2(4) = text(xtnt(1)+1.1*xtnt(3),xtnt(2)-0.1*xtnt(4),num2str(SIGMA(2,:)));
    h2(5) = text(0.2, 0.4, ['N = ', num2str(N)]);
    h2(6) = text(0.2, 0.25, ['k = ', num2str(SIGMA(2,1)/sqrt(prod(diag(SIGMA))))]);    
    set(gca,'visible','off')  
  subplot 233  
    [cs h] = contourf(x1,x2,p,6);
    cl = clevels(cs);          % The contour levels used  
    ax = axis;
    colorbar
    axis equal;
    axis(ax);  
    xlabel('x_1')
    ylabel('x_2')  
    title('True p(x_1,x_2)')
  subplot 236
    [cs h] = contourf(x(1,:),x(2,:),n,cl);
    ax = axis;
    axis equal;
    axis(ax);  
    for j=1:2
      harrow(j) =  arrow(mu(1)+[0 lambda(j,j)*U(1,j)],mu(2)+[0 lambda(j,j)*U(2,j)],0.02*diff(xlim),2,0.2,'w');
    end
    colorbar
    xlabel('x_1')
    ylabel('x_2')  
    title('Estimated p(x_1,x_2) + Eigenvectors')  


