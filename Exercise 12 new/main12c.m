%  demo of realtime signal acquisition and learning
%  Lars Kai Hansen DTU Compute 2016, with input from Sofie Therese Hansen
% 
its=25; % Number of time blocks in each signal type in pre-training
Fs = 16*1024;                    % sampling frequency in Hz
T = 0.5;                      % length of one time block in sec
%
prep_audio  % initialize audio 
%
fftstore1=zeros(its,round(0.5*nfft+1));
soundstore1=zeros(its,round(length(t)/its));
% Record "its" first type sounds
disp('Make first sound: IMPACT')
tic
for i=1:its
    recordblocking(recObj, T);
    %# get data and compute FFT
    sig = double(getaudiodata(recObj));
    % normalize
    sig=sig/std(sig);
    sigAll(sigStart:(sigStart+length(sig)-1))=sig;
    sigStart=floor((i-1)*T*Fs+1);
    sigEnd=floor(sigStart+T*Fs-1);
    fftMag = 20*log10( abs(fft(sig,nfft))+1 );
    fftstore1(i,:)=fftMag(1:round(0.5*nfft+1));
    soundstore1(i,:)=sig';
    %# update plots
    set(hLine(1), 'YData',sigAll)
    set(hLine(2), 'YData',fftMag(1:numUniq))
    title(hAx(1), num2str(i,'IMPACT  = %d'))
    drawnow                   %# force MATLAB to flush any queued displays
end
% Record "its" second type sounds
disp('Start sound: CLAP')
fftstore2=zeros(its,round(0.5*nfft+1));
soundstore2=zeros(its,round(length(t)/its));
tic
for i=1:its
    recordblocking(recObj, T);
    %# get data and compute FFT
    sig = double(getaudiodata(recObj));
    sig=sig/std(sig);
    sigAll(sigStart:(sigStart+length(sig)-1))=sig;
    sigStart=floor((i-1)*T*Fs+1);
    sigEnd=floor(sigStart+T*Fs-1);
    fftMag = 20*log10( abs(fft(sig,nfft))+1 );
    fftstore2(i,:)=fftMag(1:round(0.5*nfft+1));
    soundstore2(i,:)=sig';
    %# update plots
    set(hLine(1), 'YData',sigAll)
    set(hLine(2), 'YData',fftMag(1:numUniq))
    title(hAx(1), num2str(i,'CLAP = %d'))
    drawnow                   %# force MATLAB to flush any queued displays
end
disp('Done.')
toc
% train a linear regression
alf=0.1;
K=3;   %reduce dimensionality by PCA
Ntrain=floor(its/2);
Ntest=its-Ntrain;
% train linear regression classifier on spectra
xtrain=[fftstore1(1:Ntrain,:);fftstore2(1:Ntrain,:)];
xtest=[fftstore1((1+Ntrain):its,:);fftstore2((1+Ntrain):its,:)];
[U,S,V]=svd(xtrain',0); 
xtrain=xtrain*U(:,1:K);
xtest=xtest*U(:,1:K);
xtrain=[xtrain,ones(2*Ntrain,1)];
xtest=[xtest,ones(2*Ntest,1)];
ttrain=[ones(Ntrain,1);-ones(Ntrain,1)];
ttest=[ones(Ntest,1);-ones(Ntest,1)];
% 
D=size(xtrain,2);
w=pinv(xtrain'*xtrain+alf*eye(D))*(xtrain'*ttrain);   % w is Dx1
ttrain_pred=xtrain*w;
ttest_pred=xtest*w;
test_error=1-mean(ttest==sign(ttest_pred));
baseline=1-mean(ttest==sign(ttest(randperm(2*Ntest))));

figure
subplot(2,1,1), 
plot(1:(2*Ntrain),ttrain,'r',1:(2*Ntrain),ttrain_pred,'b')
title('TRAIN')
subplot(2,1,2)
plot(1:(2*Ntest),ttest,'r',1:(2*Ntest),ttest_pred,'b')
title(['TEST ERROR = ',num2str(test_error,2),'BASELINE ERROR = ',num2str(baseline,2)])


% run classifier realtime for second
num_secs=30;
t0=clock;
t1=t0-1;
while etime(t1,t0)<=num_secs,
     % get and classify  windows until you break out by "control c"
    recordblocking(recObj, T);
    %# get data and compute FFT
    sig = double(getaudiodata(recObj));
    sig=sig/std(sig);
    fftMag = 20*log10( abs(fft(sig,nfft))+1 );
    x=(fftMag(1:round(0.5*nfft+1)))'*U(:,1:K);
    class=sign([x,1]*w);
    if class >0,
        disp('IMPACT')
    else
        disp('CLAP')
    end
    t1=clock;
end



