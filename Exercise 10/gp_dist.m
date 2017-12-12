function dist12=gp_dist(x1,x2)
% Euclid dist between x1 and x2
% x1  d*N1
% x2  d*N2
x12=sum(x1.*x1,1);
x22=sum(x2.*x2,1);
%
dist12=x12'*ones(1,size(x2,2))+ones(size(x1,2),1)*x22 - 2*x1'*x2;
