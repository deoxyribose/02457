% MATLAB program for exercise 4 in course 02457
% This program is for part 1 out of 3 
%
% "main4a" illustrates the dependence of the
% generalization error (test error) for a linear model 
% The test error is plotted for two different models as function
% of training set size
% 
% The parameters that should be changed are
%   w_t        : the true weight-vector used to generate training-set
%   noiselevel : the Variance of the gaussian noise on training-set
%
%   (c) Lars Kai Hansen September 1999&2009

warning off
clear
close all
randn('state',0)   %fix random generator seed

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_t = [1 2 0.5]';	 % True weights
noiselevel = 0.75;   % Standard deviation of Gaussian noise on data
d = size(w_t,1);     % Number of dimensions
Nmin=5;              % Minimal training set size
Nmax=14;             % Maximal training set size
Ntest= 10000;        % Size of test set 
repetitions=10;      % number of repetitions


%%%%%%%%%% Make statistical sample of test errors for different N %%%%%%%%%%

for j=1:repetitions
 disp(['Repetition ',int2str(j),' of ',int2str(repetitions),' repetitions'])

    % d-dimensional model data set
    X1test=randn(Ntest,d);
    X1test(:,1)=ones(Ntest,1);
    Ttest=(X1test*w_t);
    noisetest = randn(Ntest,1) * noiselevel;
    Ttest= Ttest + noisetest;
    % Small model (d-1) dimensional
    X2test=X1test(:,1:(d-1));
    XX1=randn(Nmax,d);
    XX1(:,1)=ones(Nmax,1);
    TT = (XX1*w_t);
    noise = randn(Nmax,1) * noiselevel;
    TT = TT + noise;
    XX2=XX1(:,1:(d-1));

    n=1;  % counter

    for N=Nmin:Nmax

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Pick the first N  input vectors in 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X1=XX1(1:N,:);
        X2=XX2(1:N,:);
        % and the corresponding targets
        T=TT(1:N);

        % Find optimal weights for the two models
        w1 = pinv(X1)*T;
        w2=  pinv(X2)*T;

        % compute training set predictions
        Y1 = X1*w1;
        Y2 = X2*w2;

        % compute training error
        err1 = mean((Y1-T).^2);
        err2 = mean((Y2-T).^2);

        % compute test set predictions
        Y1test = X1test*w1;
        Y2test = X2test*w2;
        err1test = mean((Y1test-Ttest).^2);
        err2test = mean((Y2test-Ttest).^2);
        % save the results for later 
        test1(j,n)=err1test;
        test2(j,n)=err2test;
        train1(j,n)=err1;
        train2(j,n)=err2;
        Ns(n)=N;

        n=n+1;

    end  % end of loop over training set sizes
end  % end of repetitions

%%%%%%%%%%%%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%
figure(1), 
hold off,
h1=errorbar(Ns,mean(train1),std(train1)/sqrt(repetitions),'r:');
hold on
h2=errorbar(Ns,mean(train2),std(train2)/sqrt(repetitions),'b:');
h3=errorbar(Ns,mean(test1),std(test1)/sqrt(repetitions),'r');
h4=errorbar(Ns,mean(test2),std(test2)/sqrt(repetitions),'b');
legend('train 1', 'train 2','test 1', 'test 2')
xlabel('training set size')
ylabel('mean square errors (test and training)')
grid on
