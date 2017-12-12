%main8b.m
% RADIAL BASIS FUNCTION  CLASSIFIER
%
%  (c) Lars Kai Hansen 1999

clear 
close all

%
  K=3;           % Number of clusters  in each class
  nits=30;       % Number of EM iterations
  method=2;      % Method of initialization 1,2,3
  common_sigs=1;   % =1 if clusters have common variances , =0 if not
  close all
 seed=0;

  
% First, get some pima data...
  load pima
  meanX = mean(X_tr);
  stdX  = std(X_tr);
  % Normalize training data
  xtrain = (X_tr - repmat(meanX, size(X_tr, 1), 1)) ...
  ./ repmat(stdX, size(X_tr, 1), 1);
  ttrain = y_tr;
  % Normalize test data
  xtest  = (X_te - repmat(meanX, size(X_te, 1), 1)) ...
      ./ repmat(stdX, size(X_te, 1), 1);
  ttest  = y_te;

% Find number of classes
 C = max(y_tr);
 [Ntrain,D]=size(xtrain);
 [Ntest,D]=size(xtest);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% estimate densities of the individual classes  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% centers
yarr=zeros(C*K,D);
% variances
varr=zeros(C*K,1);
% cluster component probabilities
pkarr=zeros(C*K,1);

count=1;
% loop over classes
for c=1:C,
   % extract the relevant training and test cases
   indx=find(ttrain==c);
   ztrain=xtrain(indx,:);
   indx=find(ttest==c);
   ztest=xtest(indx,:);
   [Ntrain,D]=size(ztrain);
   [Ntest,D]=size(ztest);
      % Initialize cluster centers
   [y,sig2,prob_k]=gm_init(ztrain,K,method,0);
   % square input data
   z2train=ones(K,1)*sum((ztrain.*ztrain)');
   % plot data points
   figure(1), h_train = plot(ztrain(:,D-5),ztrain(:,D),'.');
    hold on
   h_test = plot(ztest(:,D-5),ztest(:,D),'c.');
   for k=1:K,
      h_init = plot(y(k,D-5),y(k,D),'g*'); text(y(k,D-5),y(k,D),[int2str(k),'-',int2str(0)]),drawnow
   end
   for t=1:nits,
      dist=sum((y.*y)')'*ones(1,Ntrain) + z2train -2*y*ztrain';   
      prob_z_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist);
      prob_z=sum(diag(prob_k)*prob_z_k);
      for k=1:K
         prob_k_z(k,:)=prob_k(k)*prob_z_k(k,:)./prob_z;
      end,
      y=diag(1./sum(prob_k_z'))*prob_k_z*ztrain;
      dist=sum((y.*y)')'*ones(1,Ntrain) + z2train -2*y*ztrain';   
      sig2=(1/D)*diag(1./sum(prob_k_z'))*(sum((dist.*prob_k_z)')');
      if common_sigs==1, % same sig2's
        sig2=ones(K,1)*mean(sig2);
      end
      sig_arr(:,t)=sig2;
      prob_k=sum(prob_k_z')/Ntrain;
      Etrain_arr(t)=gm_cost(ztrain,y,sig2,prob_k);
      Etest_arr(t)=gm_cost(ztest,y,sig2,prob_k);
      % plot centers
      if rem(t,15)==0,
          figure(1)
          for k=1:K,
           h_ite = plot(y(k,D-5),y(k,D),'r*'); text(y(k,D-5),y(k,D),[int2str(k),'-',int2str(t)]),drawnow
          end
          xlabel('Pima input 2 (glucose)')
          ylabel('Pima input 7 (age)')
          figure(2), 
          subplot(2,1,1),plot(sig_arr'),title(['Convergence of var. par. class No ',int2str(c)]), drawnow
          xlabel('iterations')
          subplot(2,1,2),plot(1:t,Etrain_arr,'b'),hold on,plot(1:t,Etest_arr,'r'),
          xlabel('iterations')
          hold off, title('Training (blue) and Test (red) errors '),
          drawnow
      end,
  end   %end EM
  figure(1),legend([h_train, h_test,h_init,h_ite],'train','test','\mu_k^{init}','\mu_k^{ite}')
  
  yarr(((c-1)*K+1):(c*K),:)=y;
  varr(((c-1)*K+1):(c*K))=sig2;
  pkarr(((c-1)*K+1):(c*K))=prob_k;
  clear prob_k_z prob_k prob_z  Etrain_arr Etest_arr

  count=count+1;
end %end of estimating class density

[Ntrain,D]=size(xtrain);

% estimate the prior probabilities
for c=1:C,
 P(c)=sum(ttrain==c)/Ntrain;    
end
% compute conditional densities p(x|c) for test set
[Ntest,D]=size(xtest);
x2test=ones(K,1)*sum((xtest.*xtest)');
for c=1:C,
  y=yarr(((c-1)*K+1):(c*K),:);
  sig2=varr(((c-1)*K+1):(c*K));
  prob_k=pkarr(((c-1)*K+1):(c*K));
  dist=sum((y.*y)')'*ones(1,Ntest) + x2test -2*y*xtest';   
  prob_z_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist);
  prob_z=sum(diag(prob_k)*prob_z_k);
  px_c(:,c)= prob_z';
  Pc_x(:,c)=px_c(:,c)*P(c);
end
% compute conditional class P(c|x)
px=sum(Pc_x')';
for c=1:C,
   Pc_x(:,c)=Pc_x(:,c)./px;
end
s1(1)='g';
s1(2)='b';
%plot the training points
figure(3), 
  indx1=find(ttrain==1);
  text(xtrain(indx1,D-5),xtrain(indx1,D),int2str(1)),hold on, 
  indx1=find(ttrain==2);
  text(xtrain(indx1,D-5),xtrain(indx1,D),int2str(2)),hold on, 
 
for c=1:C
% plot the density estimtes
  for k=1:K,
     plot(yarr((c-1)*K+k,D-5),yarr((c-1)*K+k,D),s1(c)),
     plot(yarr((c-1)*K+k,D-5)+sqrt(varr((c-1)*K+k))*sin(2*pi*(0:31)/30),...
        yarr((c-1)*K+k,D)+sqrt(varr((c-1)*K+k))*cos(2*pi*(0:31)/30),s1(c))
  end
end
xlabel('Pima input 2 (glucose)')
ylabel('Pima input 7 (age)')
title('Class 1 density (green), Class 2 density (blue)')
% compute the error on the test set
indx1=find(ttest==1);
indx2=find(ttest==2);
Etest11=sum(Pc_x(indx1,1) < 0.5)/length(indx1);
Etest22=sum(Pc_x(indx2,2) < 0.5)/length(indx2);
Etest=Etest11*P(1)+Etest22*P(2);
disp(['Two class pima indian problem: density estimate with K = ',int2str(K)])
disp(['The test set classification error rate is ',num2str(Etest)])
axis([-2 4 -3 5])
