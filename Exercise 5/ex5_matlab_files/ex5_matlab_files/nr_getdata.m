function [tr_i,tr_t,te_i,te_t] = getdata(Ni,t_Nh,No,ptrain,ptest,noise)
%NR_GETDATA    Create input and output data from a teacher network
%    [tr_i,tr_t,te_i,te_t] = getdata(Ni,t_Nh,No,ptrain,ptest,noise)
%    creates input and output data from a 'teacher' network. The
%    outputs are contaminated with additive white noise.
%
%    Inputs:
%         Ni     :  Number of external inputs to net
%         t_Nh   :  Number of hidden units for the 'teacher' net
%         No     :  Number of output units
%         ptrain :  Number of training examples
%         ptest  :  Number of test examples
%         noise  :  Relative amplitude of additive noise
%    Outputs:
%         tr_i, te_i :  Inputs for training & test set
%         tr_t, te_t :  Target values
%
%    See also NR_NETPRUN
%
%    Neural Regression toolbox

  % Initialize 'teacher' weights
  TWi=randn(t_Nh,Ni+1);    % Input to hidden
  TWo=randn(No,t_Nh+1);    % Hidden to output

  % Create random inputs
  tr_i = randn(ptrain,Ni);
  te_i = randn(ptest,Ni);
  
  % Determine 'teacher' outputs 
  tr_t = [tanh([tr_i ones(ptrain,1)]*TWi') ones(ptrain,1)]*TWo';
  te_t = [tanh([te_i ones(ptest,1)]*TWi') ones(ptest,1)]*TWo';

  % Add noise to each output unit column
  amp = std(tr_t);
  for u=1:No
    tr_t(:,u) = tr_t(:,u) + noise*amp(u)*randn(ptrain,1);
    te_t(:,u) = te_t(:,u) + noise*amp(u)*randn(ptest,1);
  end


