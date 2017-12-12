% MATLAB program for exercise 3 in course 02457
% This program is for part 2 out of 3 
%
% "main3b" illustrates the use of a linear model in a single 
% layer network to model the number of sunspots.
% 
% The parameters that should be changed are
%   d : The number of dimensions of the training-set.

%   (c) Karam Sidaros, September 1999.
%  Uses 
%


%%%%%%%%%%%%%%%%%%%%%%%%% Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Linear Models %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = 150;             % Number of dimensions
S = load('sp.dat'); % Load sunspot data-set
year = S(:,1);  
S = S(:,2);

d
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(S)-d;
T = S(d+1:length(S));
X = ones(N,1);
for a = 1:N
  X(a,2:d+1) = S(a:a+d-1)';
end

w = pinv(X)*T;
%w = inv(X'*X)*X'*T

Y = X*w;
err = mean((Y-T).^2);
%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%
disp('The calculated weight-vector is ');
w
disp('The training error is');
err

%%%%%%%%%%%%%%%%%%% Plotting 3D %%%%%%%%%%%%%%%%%%%

figure(1)
clf
h=get(gcf,'Position');
h0 = get(0,'defaultfigureposition');
h(3:4)=[1 1.5].*h0(3:4);
set(gcf,'Position',h);
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperType','a4');
set(gcf,'PaperOrientation','portrait');
set(gcf,'PaperPosition',[1 4 19 21]);

subplot 211
  plot(year,S,'r--',year(d+1:N+d),Y,'b-');
  ylim([0 1]);
  xlabel('Year');
  ylabel('# sunspots');
  legend('Measured','Predicted',2);

subplot 212
  plot(year(d+1:N+d),(Y-T).^2,'b-');
  xlabel('Year');
  ylabel('(y(x^n)-t^n)^2')
  title(['d = ',num2str(d),',  N = ',num2str(N),',  mean square error = ',num2str(err)]);





