% MATLAC exercise main9c.m
%
d=11;  % input dimension of sunspot prediction 
[x,tr_t,xtest,te_t] = getsun(d);
var=std([tr_t',te_t'])^2;
%
Kmax=150;
alpha=0.001;
Ntest=length(te_t);
for K=1:Kmax
    ypred=knn_regress_demo(x,tr_t,K,xtest,alpha);
    Error(K)=sum((ypred-te_t).^2)/(Ntest*var);
end
figure(1)
plot(1:Kmax,Error,'o',1:Kmax,Error,'-')
xlabel('NEAREST NEIGHBORS K')
ylabel('TEST ERROR'),grid
axis([0 Kmax 0 1])
[dummy Kopt]=min(Error);
figure(2)
ypred=knn_regress_demo(x,tr_t,Kopt,xtest,alpha);
plot(1920+(1:Ntest),te_t,'r-',1920+(1:Ntest),ypred,'b-',1920+(1:Ntest),te_t,'ro',1920+(1:Ntest),ypred,'bo')
grid, xlabel('YEAR'), ylabel('SUN SPOT INTENSITY')