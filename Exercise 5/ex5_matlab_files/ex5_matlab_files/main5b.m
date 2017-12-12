%  MATLAB main program for neural network training
%
%  The network is trained using gradient methode.
%  Sunspot series is used as dataset.

clear
d=5;                     % Sunspot input window 
Ni = d;                  % Number of external inputs
Nh = 8;                  % Number of hidden units
No = 1;                  % Number of output units
alpha_i = 0.0;           % Input weight decay
alpha_o = 0.0;           % Output weight decay
eta = 0.005;             % Learnstep size
greps = 1e-3;            % Gradient norm stopping criteria
range = 0.1;             % Initial weight range                
max_iter=2000;           % maximum number of iterations

plotfig=0;               % Plot figures while iterating

randn('seed',sum(100*clock));

% First, get some data...
[train_inp,train_tar,test_inp,test_tar] = getsun(d);
ptrain = length(train_inp);             % Number of training examples
ptest = length(test_inp);               % Number of test examples

% compute the signal variance for normalization;
sigvar=train_tar-ones(ptrain,1)*mean(train_tar);
sigvar=sum(sum(sigvar.*sigvar))/ptrain;
errnorm=2/sigvar;

% Initialize network weights
Wi = range * randn(Nh,Ni+1);
Wo = range * randn(No,Nh+1);

iter = 1;
Eold=[];
Ediff=[];
while iter < max_iter
  % Get gradient
  [dWi,dWo] =  gradient(Wi,Wo,alpha_i,alpha_o,train_inp,train_tar);
  
  % Update weights
  Wi = Wi - eta * dWi;
  Wo = Wo - eta * dWo;
  
  % Calc error and gradient
  Gradient(iter)= two_norm(dWi,dWo);
  Etrain(iter) = errnorm*cost_e(Wi,Wo,train_inp,train_tar)/ptrain;
  Etest(iter) = errnorm*cost_e(Wi,Wo,test_inp,test_tar)/ptest;
  Ediff=[Ediff abs((Eold-Etrain(end)))];
  Eold = Etrain(end);
  
  iter = iter + 1;
  
  % Plot Costfunction
  if plotfig~=0 | rem(iter,100) == 0,
    figure(2)
    semilogy(1:length(Etrain),Etrain,1:length(Etest),Etest,'g:')
    legend('Train','Test')
    ylabel('cost')
    xlabel('iterations')
    
    figure(3)
    subplot(2,1,1), title('Two-norm of gradient')
    semilogy(1:length(Gradient),Gradient,'b')
    ylabel('norm gradient')
    
    subplot(2,1,2), title('Cost function decrement')
    semilogy(1:length(Ediff),Ediff,'b')
    ylabel('train cost diff')
    xlabel('iterations')
    drawnow
    figure(4)
    [Vj,yj] = forward(Wi,Wo,test_inp);
    plot(1:length(test_tar),yj,'r',1:length(test_tar),test_tar,'b')

  end
end



