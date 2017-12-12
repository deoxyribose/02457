%  main10e  MATLAB main program for Kernel methods
%   Course 02457, November 2012, LKH
%   SVM applied to Pima indian diabetes problem
%
clear, close all
%
load pima
% whos
%   Name        Size            Bytes  Class     Attributes
% 
%   X_te      332x7             18592  double              
%   X_tr      200x7             11200  double              
%   y_te      332x1              2656  double              
%   y_tr      200x1              1600  double   
mu_tr=mean(X_tr,1);
std_tr=std(X_tr,[],1);
X_te=(X_te-ones(size(X_te,1),1)*mu_tr)./repmat(std_tr,size(X_te,1),1);
X_tr=(X_tr-ones(size(X_tr,1),1)*mu_tr)./repmat(std_tr,size(X_tr,1),1);
%X_te=X_te(:,[2,7]);
%X_tr=X_tr(:,[2,7]);
%
sig2_min=0.001;
sig2_max=20;
Nsig2=10;
C_max=5;
C_min=-1;
NCs=20;
C_array=logspace(C_min,C_max,NCs);
sig2_array=linspace(sig2_min,sig2_max,Nsig2);
%
best_ever=0;
for pp=1:Nsig2,
    disp(['Doing scale ',int2str(pp),' of ',int2str(Nsig2)])
    sig2=sig2_array(pp);
    for cc=1:NCs,
        C=C_array(cc);
        SVM = svm_train(X_tr,sign(y_tr-1.5),sqrt(sig2),C);
        % Classify test data
        [predclass]=svm_classify(X_te,SVM);
        accuracy(cc,pp)=sum(sign(y_te-1.5)==predclass)/length(y_te);
        if accuracy(cc,pp)>best_ever,
            C_best=C;
            sig2_best=sig2;
            best_ever=accuracy(cc,pp);
            SVM_best=SVM;
            pred_best=predclass;
        end
    end
end
%
figure(1)
dim1=1;
dim2=2;
indx_neg=find(y_te==1);
indx_pos=find(y_te==2);
plot(X_te(indx_neg,dim1),X_te(indx_neg,dim2),'b*',X_te(indx_pos,dim1),X_te(indx_pos,dim2),'r+')
indx_neg=find(pred_best==-1);
indx_pos=find(pred_best==1);
hold on
plot(X_te(indx_neg,dim1),X_te(indx_neg,dim2),'bo',X_te(indx_pos,dim1),X_te(indx_pos,dim2),'ro')
hold off
title(['Best test accuracy = ',num2str(best_ever),', C = ',num2str(C_best),', \sigma^2 = ',num2str(sig2)])
ss=cell(7,1);
ss{1}= 'Number of pregnancies';
ss{2}= 'Plasma glucose concentration';
ss{3}=  'Diastolic blood presure';
ss{4}=  'Triceps skin fold thickness';
ss{5}=  'Body mass index (weight/height$^2$)';
ss{6}=  'Diabetes pedigree function';
ss{7}=  'Age';
xlabel(ss{dim1})
ylabel(ss{dim2})
%
figure(2),
N_tr=size(X_tr,1);
bar(1:N_tr,sign(y_tr-1.5))
hold on,
alphas=zeros(N_tr,1);
alphas(SVM_best.index)=SVM_best.alphaHat;
plot(1:N_tr,alphas,'o')
hold off


