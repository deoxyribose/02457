function [gplog_test,pred_test_t,std_pred_test_t]=gp_loglik(test_dist,test_t,test_train_dist,train_dist,train_t,sig2,beta)
%
% 
train_N=size(train_dist,1);
test_N=size(test_dist,1);
%
A=exp(-train_dist/(2*sig2))+(1/beta)*eye(train_N);
B=exp(-test_dist/(2*sig2))+(1/beta)*eye(test_N);
C=exp(-test_train_dist/(2*sig2));
Q=C*pinv(A);
Btt=B-Q*C';
pred_test_t=Q*train_t;
gplog_test=-0.5*log(det(Btt)) -0.5*(test_t-pred_test_t)'*pinv(Btt)*(test_t-pred_test_t);
std_pred_test_t=sqrt(abs(diag(Btt)));







