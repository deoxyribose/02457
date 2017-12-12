%  main12a.m for illutration of a non-stationary linear model
%  Lars Kai Hansen, DTU Copmute 2016
%
% non-stationary linear model 
% fixed prior and noise level
d=7;
alf0=100.0;  % large precidion => small changes in time 
beta0=1.0;   % precision of additive noise
Ntrain=1000;
Ntest=1000;
% generate data
xtrain=randn(d+1,Ntrain);
xtrain(d+1,:)=ones(1,Ntrain);
ttrain=zeros(1,Ntrain);
ttest=zeros(1,Ntest);
xtest=randn(d+1,Ntest);
xtest(d+1,:)=ones(1,Ntest);
w0train=zeros(d+1,Ntrain);
w0test=zeros(d+1,Ntest);
wtrain=zeros(d+1,Ntrain);
wtest=zeros(d+1,Ntest);
% generate training and test set weights and target observations
for n=2:Ntrain,
    w0train(:,n)=w0train(:,n-1)+(1/sqrt(alf0))*randn(d+1,1);
    ttrain(n)=w0train(:,n)'*xtrain(:,n)+randn;
end
for n=2:Ntest,
    w0test(:,n)=w0test(:,n-1)+(1/sqrt(alf0))*randn(d+1,1);
    ttest(n)=w0test(:,n)'*xtest(:,n)+randn;
end

figure(1)
subplot(2,1,1),
plot(1:Ntrain,sqrt(sum(w0train.*w0train,1)),'r')
title('TRAIN WEIGHTS')
ylabel('|{\bf w}|')
xlabel('TIME')
subplot(2,1,2),
plot(1:Ntest,sqrt(sum(w0test.*w0test,1)),'b')
title('TEST WEIGHTS')
ylabel('|{\bf w}|')
xlabel('TIME')
drawnow
%  scan alpha, beta for training set modeling
alf_max=200.0;
alf_min=1;
beta_max=5;
beta_min=0.1;
Nalfs=25;
Nbetas=20;
% linear hyperarrays
alf_array=linspace(alf_min,alf_max,Nalfs);
beta_array=linspace(beta_min,beta_max,Nbetas);
% alf_array=logspace(log10(alf_min),log10(alf_max),Nalfs);
% beta_array=logspace(log10(beta_min),log10(beta_max),Nbetas);
for na=1:Nalfs,
    disp(['na = ',int2str(na),', of ',int2str(Nalfs)])
    for nb=1:Nbetas,
        alf=alf_array(na);
        beta=beta_array(nb);
        % now estimate by forward pass - cold start
        mu=zeros(d+1,1);
        Sigma=(1/alf)*eye(d+1);
        %
        for n=1:Ntrain,
            ttrain_pred(n)=mu'*xtrain(:,n);
            [ mu,Sigma] = posterior_update_linear(xtrain(:,n),ttrain(n),mu,Sigma,alf,beta);
            ttrain_est(n)=mu'*xtrain(:,n);
        end
        %
        pred_error(na,nb)=mean((ttrain-ttrain_pred).^2)/mean((ttrain).^2);
        train_error(na,nb)=mean((ttrain-ttrain_est).^2)/mean((ttrain).^2);
    end
end
[q1,q2]=find(pred_error==min(min(pred_error)));
[z1,z2]=find(train_error==min(min(train_error)));
%

% no apply to test set
%
alf=alf_array(q1);
beta=beta_array(q2);
% now estimate by forward pass
% cold start
mu=zeros(d+1,1);
Sigma=(1/alf)*eye(d+1);
mu_array=zeros(d+1,Ntest);
%
for n=1:Ntest,
    ttest_pred(n)=mu'*xtest(:,n);
    [ mu,Sigma] = posterior_update_linear(xtest(:,n),ttest(n),mu,Sigma,alf,beta);
    mu_array(:,n)=mu;
end

% 
% 
figure(2)
subplot(2,2,1),
imagesc(alf_array,beta_array,pred_error'), colorbar
hold on
plot(alf_array(q1),beta_array(q2),'w*',alf0,beta0,'r*')
title(['MIN RELATIVE PRED ERROR =',num2str(min(min(pred_error)),2)])
xlabel('\alpha')
ylabel('\beta')
%
subplot(2,2,2)
imagesc(alf_array,beta_array,train_error'), colorbar
hold on
plot(alf_array(z1),beta_array(z2),'w*')
title(['RELATIVE TRAIN ERROR=',num2str(min(min(train_error)),2)])
xlabel('\alpha')
ylabel('\beta')
% 
subplot(2,2,3)
plot(ttest,ttest_pred,'.r')
xlabel('TEST SET')
ylabel('MU TEST')
grid
title(['RELATIVE TEST ERROR =',num2str( mean((ttest-ttest_pred).^2)/mean((ttest).^2),2)])
subplot(2,2,4)
plot(1:Ntest,ttest,'b',1:Ntest,ttest_pred,'r')
xlabel('TIME')
grid

