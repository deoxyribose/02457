% main8a.m  demonstration of Gaussian Mixtures 
% for regression
%  trained by EM on 2D data
%
% (c) Lars Kai Hansen 1999
%
  clear 
  K=5;            % Number of clusters  
  nits=30;         % Number of EM iterations
  method=1;        % Method of initialization 1,2,3
  common_sigs=1;   % =1 if clusters have common variances , =0 if not
  plot_motion=1;    % plot position of center while learning (time!)
  close all
  randn('seed',0)
%
%
% getdata from sunspot file
[xtrain,ttrain,xtest,ttest]=getsun(3);
varsun=std([ttrain;ttest])^2;

% concatenate input and out vectors
ztrain=[xtrain,ttrain];
ztest=[xtest,ttest];

% initialize K mixture components
[y,sig2,prob_k]=gm_init(ztrain,K,method,0);

%
[Ntrain,D]=size(ztrain);
[Ntest,D]=size(ztest);

% compute the squares input data for later use
z2train=ones(K,1)*sum((ztrain.*ztrain)');

% plot data points
figure(1), h_train = plot(ztrain(:,D-1),ztrain(:,D),'.');
hold on
h_test = plot(ztest(:,D-1),ztest(:,D),'m.');
% plot mixture centers
for k=1:K,
    h_init = plot(y(k,D-1),y(k,D),'g*'); text(y(k,D-1),y(k,D),[int2str(k),'-',int2str(0)]),drawnow
end
ylabel(' y=x(k+1) sunspot activity')
xlabel('x(k) sunspot activity')
% iterate EM for nits iterations
for t=1:nits,
   % compute the squared distance between centers (y) and data (ztrain)
   dist=sum((y.*y)')'*ones(1,Ntrain) + z2train -2*y*ztrain'; 
   % compute P(z|k)  
   prob_z_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist);
   % compute P(z)
   prob_z=sum(diag(prob_k)*prob_z_k);
   % compute P(k|z)
   for k=1:K
      prob_k_z(k,:)=prob_k(k)*prob_z_k(k,:)./prob_z;
   end,
   % Update the centers
   y=diag(1./sum(prob_k_z'))*prob_k_z*ztrain;
   % update the widths
   dist=sum((y.*y)')'*ones(1,Ntrain) + z2train -2*y*ztrain';   
   sig2=(1/D)*diag(1./sum(prob_k_z'))*(sum((dist.*prob_k_z)')');
   % same sig2's for all components?
   if common_sigs==1,  
     sig2=ones(K,1)*mean(sig2);
   end
   sig_arr(:,t)=sig2;
   % Update the a priori proabilities
   prob_k=sum(prob_k_z')/Ntrain;
   % compute test and training errors of the density
   % model
   Etrain_arr(t)=gm_cost(ztrain,y,sig2,prob_k);
   Etest_arr(t)=gm_cost(ztest,y,sig2,prob_k);
   % plot centers
   if plot_motion==1 & rem(t,5)==0,
     figure(1)
     for k=1:K,
        h_ite = plot(y(k,D-1),y(k,D),'r*'); text(y(k,D-1),y(k,D),[int2str(k),'-',int2str(t)]),drawnow
     end
     
     figure(2), 
     subplot(2,1,1),plot(sig_arr'),title('Convergence of variance parameters'), drawnow
     subplot(2,1,2),plot(1:t,Etrain_arr,'b'),hold on,plot(1:t,Etest_arr,'r'),
     hold off, title('Training (blue) and Test (red) errors '),
     drawnow
   end,
end   %end EM
figure(1)
if plot_motion
    legend([h_train, h_test,h_init,h_ite],'train','test','\mu_k^{init}','\mu_k^{ite}')
else
    legend([h_train, h_test,h_init],'train','test','\mu_k^{init}')
end

% Plot centers and widths
figure(3), plot(ztrain(:,D-1),ztrain(:,D),'.'),hold on,
for k=1:K,
   plot(y(k,D-1),y(k,D),'r*'),
   plot(y(k,D-1)+sqrt(sig2(k))*sin(2*pi*(0:31)/30),   y(k,D)+sqrt(sig2(k))*cos(2*pi*(0:31)/30),'g')
end
ylabel(' y=x(k+1) sunspot activity')
xlabel('x(k) sunspot activity')
axis([0 1 0 1])
axis('square')

% Now prepare regression
yred=y(:,1:(D-1));
% square input data
x2train=ones(K,1)*sum((xtrain.*xtrain)');
x2test=ones(K,1)*sum((xtest.*xtest)');

dist=0.5*diag(1./sig2)*( sum((yred.*yred)')'*ones(1,Ntrain) + x2train -2*yred*xtrain');
edist=exp(-dist);
vp=prob_k.*y(:,D)';
% compute the conditional output mean on training set
that_train= (vp*edist)./(prob_k*edist);
% compute training error
train_error=sum((that_train -ttrain').^2);
train_error=train_error/varsun/Ntrain;

dist=0.5*diag(1./sig2)*( sum((yred.*yred)')'*ones(1,Ntest) + x2test -2*yred*xtest');
edist=exp(-dist);
vp=prob_k.*y(:,D)';
% compute the conditional output on test set
that_test= (vp*edist)./(prob_k*edist);
% compute test error
test_error=sum((that_test -ttest').^2);
test_error=test_error/varsun/Ntest;

disp(['Estimated ',int2str(K),' clusters and had training error ',...
 num2str(train_error),' and test error ',num2str(test_error)])

figure(4), subplot(2,1,2),plot(1:Ntest,that_test,'r',1:Ntest,ttest,'b')
title('sunspot test set')
subplot(2,1,1),plot(1:Ntrain,that_train,'r',1:Ntrain,ttrain,'b')
title('sunspot training set')


