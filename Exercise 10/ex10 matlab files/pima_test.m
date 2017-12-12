load pima

% whos
%   Name        Size            Bytes  Class     Attributes
% 
%   X_te      332x7             18592  double              
%   X_tr      200x7             11200  double              
%   y_te      332x1              2656  double              
%   y_tr      200x1              1600  double              


param_min=0.001;
param_max=20;
Nparams=10;
C_max=5;
C_min=0;
NCs=20;
C_array=logspace(C_min,C_max,NCs);
param_array=linspace(param_min,param_max,Nparams);
%
for pp=1:Nparams,
    for cc=1:NCs,
        C=C_array(cc);
        sigma=param_array(pp);
        %svm = svm_train(set1_train, @Kgaussian, param, C);
        SVM = svm_train(X_tr,sign(y_tr-1.5),sigma,C);

        % Classify test data
        [predclass]=svm_classify(X_te,SVM);

        accuracy(cc,pp)=sum(sign(y_te-1.5)==predclass)/length(y_te)
    end
end


imagesc(C_array,param_array,accuracy), colormap('gray'), colorbar