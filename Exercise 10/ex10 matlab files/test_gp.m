% simple GP experiment
%
%
d=5;  % input dimension of sunspot prediction 
[train_x,train_t,test_x,test_t] = getsun(d);
var=std([train_t',test_t'])^2;
%
Ntest=length(test_t);
Ngam=30;
Nsig2=20;
sig2_max=0.01;
sig2_min=0.001;
gam_max=10;
gam_min=0.1;
sig2_array=linspace(sig2_min,sig2_max,Nsig2);
gam_array=linspace(gam_min,gam_max,Ngam);
train_dist=gp_dist(train_x',train_x');
test_train_dist=gp_dist(test_x',train_x');
test_dist=gp_dist(test_x',test_x');
best_ever=-inf;
best_pred=0;
gplog=zeros(Ngam,Nsig2);
for gg=1:Ngam,
    gg
    for ss=1:Nsig2
        gam=gam_array(gg);
        sig2=sig2_array(ss);
        %=gp_loglik(train_t,train_dist,gam,sig2);
        [gplog_test,pred_test_t]=gp_loglik(test_dist,test_t,test_train_dist,train_dist,train_t,gam,sig2);
        gplog(gg,ss)=gplog_test;
        if gplog_test>best_ever;
            best_ever=gplog_test;
            best_pred=pred_test_t;
            best_sig2=sig2;
            best_gam=gam;
        end
    end
end
figure(1)
imagesc(gam_array,sig2_array,gplog')
hold on,
plot(best_gam,best_sig2,'r*')
 colormap('gray'), colorbar
 hold off
%
 figure(2)
Kopt=70;alpha=eps;
ypred=knn_regress_demo(train_x,train_t,Kopt,test_x,alpha);
plot(1920+(1:Ntest), test_t,'r-',1920+(1:Ntest),best_pred,'b-',1920+(1:Ntest), test_t,'ro',1920+(1:Ntest),...
    best_pred,'bo',1920+(1:Ntest), ypred,'g-',1920+(1:Ntest), ypred,'go')
grid, xlabel('YEAR'), ylabel('SUN SPOT INTENSITY')
%
title(['Test Error ',num2str(mean((pred_test_t-test_t).^2)/var),' Local Lin Test Error= ',num2str(mean((ypred-test_t).^2)/var)])
