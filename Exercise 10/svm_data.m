clear all; close all; clc

%Training data - circles
r = sqrt(rand(100,1)); % radius
t = 2*pi*rand(100,1); % angle
data1 = [r.*cos(t), r.*sin(t)]; % points
r2 = sqrt(3*rand(100,1)+1); % radius
t2 = 2*pi*rand(100,1); % angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

traindata = [data1;data2];
trainclass = ones(200,1);
trainclass(1:100) = -1;

%Test data
r = sqrt(rand(100,1)); % radius
t = 2*pi*rand(100,1); % angle
tstdata1 = [r.*cos(t), r.*sin(t)]; 
r2 = sqrt(3*rand(100,1)+1); % radius
t2 = 2*pi*rand(100,1); % angle
tstdata2 = [r2.*cos(t2), r2.*sin(t2)];

tstdata = [tstdata1;tstdata2];
tstclass = ones(200,1);
tstclass(1:100) = -1;


%Train SVM
sigma=1.7;
C=10e3;
SVM = svm_train(traindata,trainclass,sigma,C);

% Classify test data
[predclass]=svm_classify(tstdata,SVM);

pct_correct=sum(tstclass==predclass)/length(tstclass)*100

plot(traindata(1:100,1),traindata(1:100,2),'b.',traindata(101:200,1),traindata(101:200,2),'r.')
hold on
sv=SVM.sv;
sv_index=SVM.index;
disp(['Number of SV = ',int2str(length(sv))])
count=1
for n=1:length(sv_index);
    if sv_index(n)>100,
        plot(sv(count,1),sv(count,2),'ro')
        count=count+1;
    else
       plot(sv(count,1),sv(count,2),'bo')
       count=count+1;
    end
end






