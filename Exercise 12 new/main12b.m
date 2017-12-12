d=9;
S = load('sp.dat'); % Load years sunspot data-set
year = S(:,1);  clear S
% make sun spot linear model data sets
[tr_i,tr_t,te_i,te_t] = getsun(d);
tr_i=[tr_i,ones(size(tr_i,1),1)];
te_i=[te_i,ones(size(te_i,1),1)];
global_var=std([te_t;tr_t])^2;
Ntrain=length(tr_t);
Ntest=length(te_t);
w_glob=pinv(tr_i'*tr_i)*(tr_i'*tr_t);
Sigma_glob=pinv(((1/std(tr_t))^2)*(tr_i'*tr_i));
ytrain_glob=w_glob'*tr_i';
ytest_glob=w_glob'*te_i';
test_error_glob=mean((te_t'-ytest_glob).^2)/global_var;

alf=100;
beta=5;
yest=zeros(size(tr_t));   % train estimates
ypred=yest;             % train predictions

% "hot start" dynamics in global linear model
Sigma=Sigma_glob;
mu=w_glob;
% cold start
% Sigma=(1/alf)*eye(d+1);
% mu=sqrt(1/d)*rand(d+1,1);
w_train_array=zeros(d+1,Ntrain);
w_test_array=zeros(d+1,Ntest);
%
for n=1:Ntrain,
    ypred(n)=mu'*tr_i(n,:)';
    [ mu,Sigma] = posterior_update_linear(tr_i(n,:)',tr_t(n),mu,Sigma,alf,beta);
    w_train_array(:,n)=mu;
    yest(n)=mu'*tr_i(n,:)';
end
train_error=mean((tr_t-yest).^2)/global_var;
pred_error=mean((tr_t-ypred).^2)/global_var;
% 
% hot start on test set
Sigma=Sigma_glob;
mu=w_glob;
% % cold start
% Sigma=(1/alf)*eye(d+1);
% mu=randn(d+1,1);
% 
%
for n=1:Ntest,
     zpred(n)=mu'*te_i(n,:)';
     [ mu,Sigma] = posterior_update_linear(te_i(n,:)',te_t(n),mu,Sigma,alf,beta);
     w_test_array(:,n)=mu;
     zest(n)=mu'*te_i(n,:)';
end
dyn_test_error=mean((zpred-te_t').^2)/global_var;
% 
figure(1),
subplot(2,1,1)
plot(year(d:(Ntrain+d-1)),tr_t,'r',year(d:(Ntrain+d-1)),yest,'b',year(d:(Ntrain+d-1)),ypred,'k')
legend('SUNSPOT','DYNAMIC TRAIN','DYNAMIC TEST')
title('TRAINING SET')
grid
subplot(2,1,2)
plot(year((d+Ntrain+1):(Ntrain+Ntest+d)),te_t,'r',year((d+Ntrain+1):(Ntrain+Ntest+d)),ytest_glob','b',year((d+Ntrain+1):(Ntrain+Ntest+d)),zpred','k')
legend('TEST','GLOBAL TEST','DYNAMIC TEST')
title(['TEST SET ERROR GLOBAL = ',num2str(test_error_glob,2),', DYNAMIC = ',num2str(dyn_test_error,2)])
grid
%
figure(2),
subplot(2,1,1)
plot(year(d:(Ntrain+d-1)),sqrt(sum((w_glob*ones(1,Ntrain)-w_train_array).^2,1)),'r')
title('DEVIATION OF DYNAMIC FROM GLOBAL ON TRAINING SET')
grid
subplot(2,1,2)
plot(year((d+Ntrain+1):(Ntrain+Ntest+d)),sqrt(sum((w_glob*ones(1,Ntest)-w_test_array).^2,1)),'r')
title('DEVIATION OF DYNAMIC FROM GLOBAL ON TEST SET')
grid

