% MATLAB program for exercise 4 in course 02457
% This program is for part 3 out of 3 
%
% "main4c" illustrates the bias variance
% decomposition for a linear model 
% The bias and variance components are
% plotted for models with increasing weight decay
% This version corrected: Now uses a common testset
% as in the definition. Thanks to Lars Spicker Olsen
% The weight decay range has been changed also
%
%   (c) Lars Kai Hansen September 1999,2000&2009.
clear
close all
warning off
randn('state',0)   %fix random generator seed

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_t = [1 2 0.5]';	   % True weights
noiselevel = 1.0;   % Variance of Gaussian noise on data
d = size(w_t,1);     % Number of dimensions
N = 5;              % Training set size
Ntest= 10000;        % Size of test set 
repetitions=20;      % number of repetitions
alfmin = 0.03;        % minimal weigth decay
alfmax = 100.0;       % maximal weigth decay
M=10;                % number of weight decays

%%%%%%%% Pre-allocation of variables %%%%%%%%
alfs = NaN*ones(1,M);
meanarr = NaN*ones(1,M);
biasarr = NaN*ones(1,M);
variancearr = NaN*ones(1,M);
Ytest = NaN*ones(Ntest,repetitions);

%%%%%%%%%% Make statistical sample of samples for bias&variance %%%%%%%%%%
n=1;
for k=1:M,  % weight decays
    alf = alfmin*(alfmax/alfmin).^((k-1)/(M-1));
    disp(['Weight decay # ',int2str(k),' of ',int2str(M),' decays'])
     % d-dimensional model data set
    Xtest=randn(Ntest,d);
    Xtest(:,1)=ones(Ntest,1);
    Ttest=(Xtest*w_t);
    noisetest = randn(Ntest,1) * noiselevel;
    Ttest= Ttest + noisetest;

    for j=1:repetitions
        % Small model (d-1) dimensional
        Xtrain=randn(N,d);
        Xtrain(:,1)=ones(N,1);
        Ttrain = (Xtrain*w_t);
        noise = randn(N,1) * noiselevel;
        Ttrain = Ttrain + noise;

        % Find optimal weights for the regularized model
        w = pinv(Xtrain'*Xtrain + alf*eye(d))*(Xtrain'*Ttrain);

        % compute test set predictions
        Ytest(:,j) = Xtest*w;

        % save the results for later 

    end  % end of repetitions

    Ybias = mean(Ytest,2);
    bias_error=mean( (Ybias-Ttest).^2 );
    mean_error=0;
    for j=1:repetitions,
       mean_error = mean_error + mean((Ytest(:,j)-Ttest).^2);
    end
    mean_error=mean_error/repetitions;
    variance_error=mean_error-bias_error;  


    alfs(n)=alf;
    meanarr(n)=mean_error;
    biasarr(n)=bias_error;
    variancearr(n)=variance_error;
    n=n+1;

end  % end of weight decay loop


%%%%%%%%%%%%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%
figure(1), 
hold off,
h1=semilogx(alfs,meanarr,'r');
hold on
h2=semilogx(alfs,biasarr,'b');
h3=semilogx(alfs,variancearr,'g');
legend('mean test error', 'bias','variance')
xlabel('weight decay')
ylabel('mean square errors (test, bias and variance)')

