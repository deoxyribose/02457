% COURSE 02457 EXERCISE 11 
% Lars Kai Hansen (c) 2016
%
% NON-stationary MARKOV CHAIN ESTIMATION
% How detection of shift in markov process depends on window size
% fix seed for reproducibility  
rng(1);
% size of Markov model
K=10;
% create sequence
N1=10000;  % Markov model changes at N1, N1+N2 total sample size N1+N2+N1
N2=10000;
% Markov models
a1=10-10*eye(K)+rand(K); %+1*eye(K);
a1=a1./(sum(a1,2)*ones(1,K));
a2=rand(K)+10*eye(K);
a2=a2./(sum(a2,2)*ones(1,K));
%
% generate non-stationary Markov sequence
x=zeros(1,N1+N2+N1);
x(1)=1;
for n=1:N1
  x(n+1)=getint(cumsum(a1(x(n),:)));
end
for n=1:N2
  x(N1+n+1)=getint(cumsum(a2(x(N1+n),:)));
end
%
for n=1:(N1-1)
  x(N1+N2+n+1)=getint(cumsum(a1(x(N1+N2+n),:)));
end
%
% parameters
alf=1;   % prior: alf = alpha-1 in Dirichlet prior
skip=1;    % skip between window starts
Nwin=9;    % number of window lengths explored
win_max=5000;  % max window size
win_min=50;    % min window size
win_array=round(linspace(win_min,win_max,Nwin));
win_pred=1000;  % window for est of test likelihood
%
loglik_train=zeros(Nwin,ceil((N1+N2+N1)/win_min));
loglik_test=zeros(Nwin,ceil((N1+N2+N1)/win_min));
loglik1=zeros(Nwin,ceil((N1+N2+N1)/win_min));
loglik2=zeros(Nwin,ceil((N1+N2+N1)/win_min));
for w=1:Nwin
    disp(['Window size ',int2str(w),', of ',int2str(Nwin)])
    % random init ae's
    winsize=win_array(w);
    Nwindows=floor((N1+N2+N1-win_pred-winsize)/skip)-1;
    Nwindows_array(w)=Nwindows;
    startwin=1;
    for t=1:Nwindows
        xwin=x(startwin:(startwin+winsize-1));
        xwin_pred=x((startwin+winsize+1):(startwin+winsize+win_pred)); % predict ahead (in stream predict with old model)
        ae=markov_map(xwin,K,alf); % estimate model
        loglik_test(w,t)=markov_loglik(xwin_pred,ae,K)/win_pred;
        if (startwin<= N1)||(startwin>(N1+N2)) % L1-distance to true model
            dist_true(w,t)=sum(sum(abs(ae-a1)))/sum(sum(a1));
        else
            dist_true(w,t)=sum(sum(abs(ae-a2)))/sum(sum(a2));
        end
        loglik_train(w,t)=markov_loglik(xwin,ae,K)/win_array(w);
        loglik1(w,t)=markov_loglik(xwin,a1,K)/win_array(w);
        loglik2(w,t)=markov_loglik(xwin,a2,K)/win_array(w);
        timers(w,t)=startwin;
        startwin=startwin+skip;
    end
    mean_loglik_test(w)=mean(loglik_test(w,1:Nwindows));
end

% plot training likelihood and true models' likelihoods
figure(1),
nnn=ceil(sqrt(Nwin));
for w=1:Nwin
    subplot(nnn,nnn,w)
    
    plot(timers(w,1:Nwindows_array(w)),loglik_train(w,1:Nwindows_array(w)),'r',...
        timers(w,1:Nwindows_array(w)),loglik1(w,1:Nwindows_array(w)),'b',...
        timers(w,1:Nwindows_array(w)),loglik2(w,1:Nwindows_array(w)),'g')
    title(['WIN LENGTH = ',int2str(win_array(w))])
    if w==1,legend('TRAIN','TRUE 1','TRUE 2'),end
     xlabel('TIME'),ylabel('LOGLIK')
    grid
end


% plot training and test likelihoods
figure(2),
nnn=ceil(sqrt(Nwin));
for w=1:Nwin
    subplot(nnn,nnn,w)
    plot(timers(w,1:Nwindows_array(w))+win_array(w),loglik_test(w,1:Nwindows_array(w)),'k',timers(w,1:Nwindows_array(w)),loglik_train(w,1:Nwindows_array(w)),'r')
  title(['WIN LENGTH = ',int2str(win_array(w)),', MEAN TEST LOGLIK =',num2str(mean_loglik_test(w),3)])
    if w==1,legend('TEST','TRAIN'), end
    xlabel('TIME'),ylabel('LOGLIK')
    grid
end



% plot distance to true model 
figure(3),
nnn=ceil(sqrt(Nwin));
for w=1:Nwin
    subplot(nnn,nnn,w)
    mean_model_mse(w)=   mean(dist_true(w,:));  
    plot(timers(w,1:Nwindows_array(w)),dist_true(w,1:Nwindows_array(w)),'r')
    title(['WIN LENGTH = ',int2str(win_array(w)),', MEAN DIST = ',num2str(mean(dist_true(w,:)),3)])
    xlabel('TIME')
    ylabel('L1-DIST |EST - TRUE|')
    grid
end

% plot test set measure 
figure(4)
subplot(2,1,1)
bar(win_array,mean_model_mse)
%bar(log10(win_array),mean_model_mse)
xlim=[0 win_max];
xlabel('WINDOW SIZE')
%xlabel('LOG10 WINDOW SIZE')
ylabel('MEAN L1-DIST |EST - TRUE|')
subplot(2,1,2),
%bar(log10(win_array),mean_loglik_test)
bar(win_array,mean_loglik_test)
xlim=[0 win_max];
%xlabel('LOG10 WINDOW SIZE')
xlabel('WINDOW SIZE')
ylabel('MEAN TEST LOGLIK')

