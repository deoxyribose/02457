% MATLAB program for exercise 4 in course 02457
% This program is for part 2 out of 3 
%
% "main3b" illustrates the use test errors for selecting
% a linear model in a single layer network 
% model of the number of sunspots.
% 
% The parameters that should be changed are
%   d : The number of dimensions of the training-set.
%
%   (c) Lars Kai Hansen & Karam Sidaros, September 1999&2009.
clear,
close all
warning off

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = 9;             % Number of dimensions
S = load('sp.dat'); % Load sunspot data-set
year = S(:,1);  
S = S(:,2);

var = std(S).^2;   % the total signal variance for normalization
                   %  of variances
last_train = 221-d;                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create lag space matrix 
N = length(S)-d;
T = S(d+1:length(S));
X = ones(N,1);
for a = 1:N
  X(a,2:d+1) = S(a:a+d-1)';
end
%Training set
Xtrain=X(1:last_train,:);
Xtest=X((last_train+1):N,:);
Ttrain=T(1:last_train);
Ttest=T((last_train +1):N);


w = pinv(Xtrain)*Ttrain;
Ytrain = Xtrain*w;
Ytest = Xtest*w;
Y = X*w;
errtrain = mean((Ytrain-Ttrain).^2)/var;
errtest  = mean((Ytest-Ttest).^2)/var;
%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%
disp('The calculated weight-vector is ');
w
disp('The relative training error is');
errtrain
disp('The relative test error is');
errtest


%%%%%%%%%%%%%%%%%%% Plotting 3D %%%%%%%%%%%%%%%%%%%

figure(1)
clf
subplot 211
  plot(year,S,'r--',year(d+1:N+d),Y,'b-');
  ylim([0 1]);
  xlabel('Year');
  ylabel('# sunspots');
  legend('Measured','Predicted',2);

subplot 212
  plot(year(d+1:N+d),(Y-T).^2,'b-');
  xlabel('Year');
  ylabel('(y({\bf x}_n)-t_n)^2')
  title(['d = ',num2str(d),', rel. mean square test error = ',num2str(errtest)]);






