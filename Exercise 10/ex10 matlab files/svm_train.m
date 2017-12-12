function SVM = svm_train(x,y,par,C)
%%TRAINSVM Train a basic non-linear rbf Support Vector Machine
%
% (x,y) is the training set
% x is the pattern vector in R^n 
% y is the corresponding class labels in R^1 y={1,-1}
% C is the regularization parameter
% par is the kernel parameter, sigma

%Set parameters if empty
if isempty(par)
par=1;
end
if isempty(C)
C=1;
end

%Scale Data (can be replaced by other scaling)
shift = - mean(x);
stdVals = std(x);
scaleFactor = 1./stdVals;
% leave zero-variance data unscaled:
scaleFactor(~isfinite(scaleFactor)) = 1;

% shift and scale columns of data matrix:
for c = 1:size(x, 2)
    x(:,c) = scaleFactor(c) * (x(:,c) +  shift(c));
end

%Generate kernel
kern=svm_kernel_rbf(x,x,par);

%make kernel symmetric
kern=(kern+kern')/2; % + diag(1./(ones(length(y),1)*C));

%H: Represents the "Hessian" in the QP min 1/2*x'*H*x + f'*x
H=(y*y').*kern;

%Make H symmetric
%H=(H+H')/2;

%f: Represents the linear term in the expression min 1/2*x'*H*x + f'*x
f=-ones(length(x),1);

%Aeq: Represents the linear coefficients in the constraints Aeq*x = beq.
Aeq=y';

%beq: Represents the constant vector in the constraints Aeq*x = beq.
beq=0;


%lb: Represents the lower bounds elementwise in lb.
lb=zeros(length(y),1);

%ub: Represents the upper bounds elementwise in ub.
ub=C*ones(length(y),1);

% QP solver
%opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-7,'TolFun',1e-7,'TolCon',1e-7);
%opts = optimset('Display','off','TolX',1e-7,'TolFun',1e-7,'TolCon',1e-7);
%opts = optimset('Algorithm','interior-point-convex','Display','off');

if verLessThan('matlab', '7.12')
    opts = optimset('Algorithm','interior-point','Display','off','TolX',1e-7,'TolFun',1e-7,'TolCon',1e-7);
else   
    opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-7,'TolFun',1e-7,'TolCon',1e-7);
end

[alpha,~,~,~,lambda]  = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],opts);

% The support vectors are the non-zero of alpha.
sv_index = find(alpha > sqrt(eps));
sv = x(sv_index,:);

alphaHat = y(sv_index).*alpha(sv_index);
        
% Find the bias - several possibilities.
[~,max_pos] = max(alpha);
bias = y(max_pos) - sum(alphaHat.*kern(sv_index,max_pos));


%Rescale data
for c = 1:size(sv, 2)
        sv(:,c) = (sv(:,c)./scaleFactor(c)) - shift(c);
end

%Return SVM data structure (used for input in classifySVM.m)
SVM.sv=sv;
SVM.index=sv_index;
SVM.lambda=lambda;
SVM.alphaHat=alphaHat;
SVM.bias=bias;
SVM.shift=shift;
SVM.scaleFactor=scaleFactor;
SVM.par=par;