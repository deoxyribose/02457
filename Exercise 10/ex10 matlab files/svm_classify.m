function [predclass]=svm_classify(newdata,SVM)
%CLASSIFYSVM Predict class of test data using the SVM structure obtained
%from svm_train.m

%Unpack SVM structure
sv=SVM.sv;
alphaHat=SVM.alphaHat;
bias=SVM.bias;
shift=SVM.shift;
scaleFactor=SVM.scaleFactor;
par=SVM.par;

% shift and scale columns of data matrix:
for c = 1:size(sv, 2)
    sv(:,c) = scaleFactor(c) * (sv(:,c) +  shift(c));
end

for c = 1:size(newdata, 2)
    newdata(:,c) = scaleFactor(c) * (newdata(:,c) +  shift(c));
end

%Classify new data
f=svm_kernel_rbf(sv,newdata,par)'*alphaHat(:)+bias;
predclass = sign(f);

% points on the boundary are assigned to class 1
predclass(predclass==0) = 1;
end