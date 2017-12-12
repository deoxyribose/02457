%  main10c  MATLAB main program for Kernel pdf estimation
%   Course 02457, November 2012, LKH
%   Simple kernel GP applied to the sunspot data
%
clear, close all
%
d=5;  % input dimension of sunspot prediction 
[train_x,train_t,test_x,test_t] = getsun(d);
var=std([train_t',test_t'])^2;
%
Ntest=length(test_t);
Nsig2=20;
Nbeta=20;
beta_max=1;
beta_min=500;
sig2_max=2;
sig2_min=0.01;
beta_array=linspace(beta_min,beta_max,Nbeta);
sig2_array=linspace(sig2_min,sig2_max,Nsig2);
train_dist=gp_dist(train_x',train_x');
test_train_dist=gp_dist(test_x',train_x');
test_dist=gp_dist(test_x',test_x');
best_ever=-inf;
best_ls=inf;
gplog=zeros(Nsig2,Nbeta);
gpls=zeros(Nsig2,Nbeta);
for gg=1:Nsig2,
    disp(['Computing predictions in scale ',int2str(gg),' of ', int2str(Nsig2)])
    for ss=1:Nbeta
        sig2=sig2_array(gg);
        beta=beta_array(ss);
        [gplog_test,pred_test_t,std_pred_test_t]=gp_loglik(test_dist,test_t,test_train_dist,train_dist,train_t,sig2,beta);
        gplog(gg,ss)=gplog_test;
        if gplog_test>best_ever;
            best_ever=gplog_test;
            best_pred=pred_test_t;
            best_std_pred=std_pred_test_t;
            best_beta=beta;
            best_sig2=sig2;
        end
        ls=mean((pred_test_t - test_t).^2)/var;
        gpls(gg,ss)=ls;
        if ls < best_ls;
            best_ls=ls;
            best_pred_ls=pred_test_t;
            best_beta_ls=beta;
            best_sig2_ls=sig2;
        end
    end
end
figure(1)
subplot(2,1,1)
imagesc(sig2_array,beta_array,gplog')
hold on,
plot(best_sig2,best_beta,'r*')
 colormap('gray'), colorbar
 hold off
subplot(2,1,2)
imagesc(sig2_array,beta_array,gpls')
hold on,
plot(best_sig2_ls,best_beta_ls,'g*')
 colormap('gray'), colorbar
 hold off
 
%
 figure(2)
plot(1920+(1:Ntest), test_t,'r-',1920+(1:Ntest),best_pred,'b-',...
    1920+(1:Ntest),test_t,'ro',1920+(1:Ntest),best_pred,'bo',1920+(1:Ntest),best_pred_ls,'g-',1920+(1:Ntest),best_pred_ls,'go')
grid, xlabel('YEAR'), ylabel('SUN SPOT INTENSITY')
title(['Test Error LL ',num2str(mean((best_pred-test_t).^2)/var),' Test Error LS ',num2str(mean((best_pred_ls-test_t).^2)/var)])

% 
figure(3)
% plot GP with posterior uncertainty
plot(1920+(1:Ntest), test_t,'r-',1920+(1:Ntest),best_pred,'b-',...
    1920+(1:Ntest),test_t,'ro',1920+(1:Ntest),best_pred,'bo',1920+(1:Ntest),best_pred+2*best_std_pred,'b:',1920+(1:Ntest),best_pred-2*best_std_pred,'b:')
axis([1920 1980 0.0 1.2])
grid

