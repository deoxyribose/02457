% main7b.m  demonstration of Gaussian Mixtures 
%  trained by EM on 2D data
%
% (c) Lars Kai Hansen (1999)
% revised by Carsten Stahlhut (2009)
%
  clear 
  Ntrain=120;    % number of training examples
  Ntest=300;     % number of test examples
  noise=0.06;    % width of true clusters
  K=5;           % Number of clusters  
  D=2;           % Dimension of data
  nits=30;       % Number of EM iterations
  method=3;      % Method of initialization 1,2,3
  close all
  randn('seed',0)
%
%
% getdata
[xtrain,xtest]=getdata(Ntrain,Ntest,noise);
[y,sig2,prob_k]=gm_init(xtrain,K,method);

% square input data
x2train=ones(K,1)*sum((xtrain.*xtrain)');

% plot data points
figure(1), h_train = plot(xtrain(:,1),xtrain(:,2),'.');
hold on
h_test = plot(xtest(:,1),xtest(:,2),'m.');
for k=1:K,
      h_init = plot(y(k,1),y(k,2),'g*'); text(y(k,1),y(k,2),[int2str(k),'-',int2str(0)]),drawnow
end
for t=1:nits,
   dist=sum((y.*y)')'*ones(1,Ntrain) + x2train -2*y*xtrain';                % || x_n - mu_k ||^2
   prob_x_k=diag(1./((2*pi*sig2).^(D/2)) )*exp(-0.5*diag(1./sig2)*dist);    % p(x|mu_k,sig2_k)
   prob_x=sum(diag(prob_k)*prob_x_k);                                       % p(x|w)
   for k=1:K
      prob_k_x(k,:)=prob_k(k)*prob_x_k(k,:)./prob_x;                        % p(k|x_n) = gamma_{nk}
   end,
   y=diag(1./sum(prob_k_x'))*prob_k_x*xtrain;                               % y(k) = mu_k
   
   dist=sum((y.*y)')'*ones(1,Ntrain) + x2train -2*y*xtrain';   
   sig2=(1/D)*diag(1./sum(prob_k_x'))*(sum((dist.*prob_k_x)')');
   sig_arr(:,t)=sig2;
   prob_k=sum(prob_k_x')/Ntrain;                                            % pi_k
   Etrain_arr(t)=gm_cost(xtrain,y,sig2,prob_k);
   Etest_arr(t)=gm_cost(xtest,y,sig2,prob_k);
   % plot centers
   if rem(t,5)==0,
     figure(1)
     for k=1:K,
        h_ite = plot(y(k,1),y(k,2),'r*'); text(y(k,1),y(k,2),[int2str(k),'-',int2str(t)] );
        drawnow;
     end
     figure(2), 
     subplot(2,1,1),plot(sig_arr'),title('Convergence of variance parameters'), drawnow
     subplot(2,1,2),plot(1:t,Etrain_arr,'b'),hold on,plot(1:t,Etest_arr,'r'),
     hold off, title('Training (blue) and Test (red) errors '),
     drawnow
   end,
end   %end EM
figure(2), legend([h_train, h_test,h_init,h_ite],'train','test','\mu_k^{init}','\mu_k^{ite}')

figure(3), plot(xtrain(:,1),xtrain(:,2),'.'),hold on,
for k=1:K,
   plot(y(k,1),y(k,2),'r*'),
   plot(y(k,1)+sqrt(sig2(k))*sin(2*pi*(0:31)/30),   y(k,2)+sqrt(sig2(k))*cos(2*pi*(0:31)/30),'g')
end
axis([0 1 0 1])
axis('square')

disp(Etest_arr(end))

