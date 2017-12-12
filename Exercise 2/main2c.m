
% MATLAB program for exercise 2 in course 02457
% This program is for part 3 out of 4 
%
% "main2a" visualizes the coordinate transformation of a given data-set
% to obtain a new data-set with zero mean and a covariance');
% matrix equal to the unit matrix');
%
% This program allows you to vary the covariance matrix of a 
% 2D Normal distribution and makes contour plots of the PDF 
% and the histogram of the sample both before and after the 
% coordinate transformation.

% (c) Karam Sidaros, September 1999. 
% Uses randmvn.m mvnpdf.m hist2d.m arrow.m clevels.m
%

%%%%%%%%%%%%%%%%%%%%%%%%% Part 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Coordinate Transformation  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu=[5 2]';	    % true mean value
SIGMA=[2   -1.2 
       -1.2 3  ];   % true covariance matrix
N=10000;            % number of points in density

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  disp('The true mean is');
  mu
  disp('and the true covariance matrix is');
  SIGMA
  disp('The number of samples is');
  N
  
  D = randmvn(mu,SIGMA,N);  % A random sample - set of N samples
  
  mu_ = mean(D,2);          % Estimated mean
  SIGMA_ = cov(D');         % Estimated covariance matrix

  disp('The estimated mean is');
  mu_
  disp('and the estimated covariance matrix is');
  SIGMA_
  
  [U lambda] = eig(SIGMA_); % Eigenvectors and eigenvalues

%%%%%%%%%%%%% Coordinate Transformation %%%%%%%%%%%%%%%%%  
%  D_ = sqrt(inv(lambda))*U'*(D-repmat(mu_,1,N));  % Coordinate transformation
 D_ = U'*(D-repmat(mu_,1,N));  % Coordinate transformation

  mu_t = mean(D_,2);  % Estimated mean after transformation
  SIGMA_t = cov(D_'); % Estimated covariance matrix after transformation
  SIGMA_t = SIGMA_t .* (abs(SIGMA_t)>(100*eps)); %  Correction for rounding errors
  [U_ lambda_] = eig(SIGMA_t); % Eigenvectors and eigenvalues after transformation
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  disp('The estimated mean after transformation is');
  mu_t
  disp('and the estimated covariance matrix after transformation is');
  SIGMA_t
  
  fprintf('Calculating histogram ... ');
  nbins = 20;                   % # bins in histogram in each dimension
  [n,x] = hist2d(D,nbins,[],1);     % Histogram
  [n_ x_] = hist2d(D_,nbins,[],1);  % Histogram after coord. transformation
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
  
  range_ = x_(:,nbins)-x_(:,1);
  x1_ = x_(1,1):range_(1)/(resol-1):x_(1,nbins);
  x2_ = x_(2,1):range_(2)/(resol-1):x_(2,nbins);
  
  [X1_ X2_] = meshgrid(x1_,x2_);
  X1_=X1_(:);X2_=X2_(:);
  fprintf('Calculating true PDF ... ');
  p_ = mvnpdf([0;0],eye(2),[X1_';X2_']);% True PDF after coord. transformation
  p_ = reshape(p_,[resol resol]);
  fprintf('Done\n\r');
  
  
%%%%%%%%%%%%%%%%%%%%%%  Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
  subplot 232
    [cs h] = contourf(x1,x2,p,6);
    cl = clevels(cs);          % The contour levels used  
    ax = axis;
    colorbar
    axis equal;
    axis(ax);  
    xlabel('x_1')
    ylabel('x_2')  
    title('True PDF')
  subplot 233
    [cs h] = contourf(x(1,:),x(2,:),n,cl);
    ax = axis;
    axis equal;
    axis(ax);  
    for j=1:2
      harrow(j) =  arrow(mu_(1)+[0 lambda(j,j)*U(1,j)],mu_(2)+[0 lambda(j,j)*U(2,j)],0.04*diff(xlim),2,0.25,'w');
    end
    colorbar
    xlabel('x_1')
    ylabel('x_2')  
    title('Estimated PDF + Eigenvectors')  
  subplot 235
    [cs h] = contourf(x1_,x2_,p_,6);
    cl = clevels(cs);          % The contour levels used  
    ax = [-3 3 -3 3];
    axis equal;
    axis(ax);  
    colorbar
    xlabel('x_1~')
    ylabel('x_2~')  
    title('Transformed True PDF')
  subplot 236 
    [cs h] = contourf(x_(1,:),x_(2,:),n_,6);
    colorbar
    axis equal;
    axis(ax);  
    for j=1:2
      harrow(j) =  arrow(mu_t(1)+[0 lambda_(j,j)*U_(1,j)],mu_t(2)+[0 lambda_(j,j)*U_(2,j)],0.04*diff(xlim),2,0.25,'w');
    end
    xlabel('x_1~')
    ylabel('x_2~')  
    title('Transformed Estimated PDF + Eigenvectors')  
  subplot 234
    h2 = text(0.0, 1.0,['\mu = (', num2str(mu'), ')^T']);
    h2(2) = text(0.0,0.8,['\Sigma = ']);
    xtnt = get(h2(2),'extent');
    h2(3) = text(xtnt(1)+1.1*xtnt(3),xtnt(2)+1.0*xtnt(4),num2str(SIGMA(1,:)));
    h2(4) = text(xtnt(1)+1.1*xtnt(3),xtnt(2)-0.1*xtnt(4),num2str(SIGMA(2,:)));
    h2(5) = text(0.0, 0.6, ['N = ', num2str(N)]);
    h2(6) = text(0.0, 0.45, ['k = ',  num2str(SIGMA(2,1)/sqrt(prod(diag(SIGMA))))]); 
    h2(7) = text(0.0, 0.3, ['\lambda_1 = ', num2str(lambda(1,1),3)]);
    h2(8) = text(0.65, 0.3, ['\lambda_2 = ', num2str(lambda(2,2),3)]);
    h2(9) = text(0.0, 0.15,['u_1 = (', num2str(U(:,1)',2), ')^T']);
    h2(10) = text(0.65, 0.15,['u_2 = (', num2str(U(:,2)',2), ')^T']);
    
    set(gca,'visible','off')  
  
