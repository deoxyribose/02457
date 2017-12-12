% MATLAB program for exercise 2 in course 02457
% This program is for part 4 out of 4 
%
% "main2d" visualizes the dimensionality reduction through
%  projection on the eigenvectors of the covariance matrix.
%
% This program plots the data from a 2D classification problem
% along with the marginal histograms both before and after the
% coordinate transformation.

% (c) Karam Sidaros, September 1999.
% Uses randmvn.m mvnpdf.m randbin.m arrow.m
%

%%%%%%%%%%%%%%%%%%%%%%%%% Part 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Projection on Eigenvectors %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off
%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu1=0.8*[2 3]';	    % true mean value
SIGMA1=[1.2  1 
       1 4  ];   % true covariance matrix

mu2=-1.1*[2 3]';	    % true mean value
SIGMA2=[2   -1.4 
       -1.4 2  ];   % true covariance matrix

p1 = 0.3;          % P(C1)  probabilty of class 1
p2 = 1-p1;

N=10000;            % number of points in density


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D1 = randmvn(mu1,SIGMA1,N);  % A random sample
D2 = randmvn(mu2,SIGMA2,N);  % A random sample

c1 = randbin(p1,1,N);        % Occurrence of class 1
c2 = 1 - c1;                 % Occurrence of class 2

D  = D1.*[c1;c1] + D2.*[c2; c2];

mu = mean(D,2);          % Estimated mean
SIGMA = cov(D');         % Estimated covariance matrix

disp('The estimated mean is');
mu
disp('and the estimated covariance matrix is');
SIGMA
  
[U lambda] = eig(SIGMA); % Eigenvectors and eigenvalues

%%%%%%%%%%%%% Coordinate Transformation %%%%%%%%%%%%%%%%%  
%  D_ = sqrt(inv(lambda))*U'*(D-repmat(mu,1,N));  % Coordinate transformation
  D_ = U'*(D-repmat(mu,1,N));  % Coordinate transformation

  mu_ = mean(D_,2);  % Estimated mean after transformation
  SIGMA_ = cov(D_'); % Estimated covariance matrix after transformation
  SIGMA_ = SIGMA_ .* (abs(SIGMA_)>(100*eps)); %  Correction for rounding errors
  [U_ lambda_] = eig(SIGMA_); % Eigenvectors and eigenvalues after transformation
  
%%%%%%%%%%%%%%%%% Marginal Histograms %%%%%%%%%%%%%%%%%%%%%%%%

  fprintf('Calculating histograms ... ');
  nbins = 50;                   % # bins in histogram 
  [n1 x1] = hist(D(1,:),nbins); % Marginal Histograms
  [n2 x2] = hist(D(2,:),nbins);
  [n1_ x1_] = hist(D_(1,:),nbins);
  [n2_ x2_] = hist(D_(2,:),nbins);
  n1 = n1/sum(n1);  
  n2 = n2/sum(n2);  
  n1_ = n1_/sum(n1_);  
  n2_ = n2_/sum(n2_);  
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
    for j=1:2
      harrow(j) =  arrow(mu(1)+[0 lambda(j,j)*U(1,j)],mu(2)+[0 lambda(j,j)*U(2,j)],0.04*diff(xlim),2,0.25,'w');
    end
    xlabel('x_1')
    ylabel('x_2')  
    title('Scatter plot of data-set');
  subplot 232
    barh(x2,n2,1,'w');
    pos = get(gca,'position');
    pos([2 4]) = h([2 4]);
    set(gca,'position',pos);    
    set(gca,'plotboxaspectratio',h1);
    ylim(ax(3:4));
    xlim([0 max(n2)*1.02]);    
    ylabel('x_2')  
    xlabel('p(x_2)')
    title('Marginal Distribution');    
  subplot 234
    bar(x1,n1,1,'w');
    pos = get(gca,'position');
    pos([1 3]) = h([1 3]);
    set(gca,'position',pos);    
    set(gca,'plotboxaspectratio',h1);
    xlim(ax(1:2));
    ylim([0 max(n1)*1.02]);    
    xlabel('x_1')
    ylabel('p(x_1)')
    title('Marginal Distribution');    
  subplot 236 
    plot(D_(1,:),D_(2,:),'.'); 
    ax=axis;
    axis equal;
    axis(ax);
    h = get(gca,'position');
    h1 = get(gca,'plotboxaspectratio');   
    for j=1:2
      harrow(j) =  arrow(mu_(1)+[0 lambda_(j,j)*U_(1,j)],mu_(2)+[0 lambda_(j,j)*U_(2,j)],0.04*diff(xlim),2,0.25,'w');
    end
    xlabel('^~x_1')
    ylabel('^~x_2')  
    title('Scatter plot of transf. data-set ');
  subplot 233
    bar(x1_,n1_,1,'w');
    pos = get(gca,'position');
    pos([1 3]) = h([1 3]);
    set(gca,'position',pos);    
    set(gca,'plotboxaspectratio',h1);
    xlim(ax(1:2));
    ylim([0 max(n1_)*1.02]);    
    xlabel('^~x_1')
    ylabel('p(^~x_1)')
    title('Marginal Distribution after transf.');    
  subplot 235
    barh(x2_,-n2_,1,'w');
    pos = get(gca,'position');
    pos([2 4]) = h([2 4]);
    set(gca,'position',pos);    
    set(gca,'plotboxaspectratio',h1);
    ylim(ax(3:4));
    xlim([-max(n2_)*1.02 0]);    
    set(gca,'xticklabel',num2str(-str2num(get(gca,'xticklabel'))));
    ylabel('^~x_2')  
    xlabel('p(^~x_2)')
    title('Marginal Distribution after transf.');    
  axes('position',[0 0 1 1],'units','normalized');
  h = plot([.33 .33 .67 .67],[0 .5 .5 1],'k-');
  set(h,'linewidth',5);  
  axis([0 1 0 1]);  
  set(gca,'visible','off')  


return
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
  
