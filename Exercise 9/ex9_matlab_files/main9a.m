%  main9.a  MATLAB main program for Kernel pdf estimation
%   Course 02457, November 2007, LKH
%  
clear
%
Nx=50;          % dimensions of 2D data 
Ny=50;
N=300;          % training set size
Nval=200;       % test set size
width=10;       % std of simulation normal distribution
%
% Normal distributed data for training set
xtrain=(Nx/2)+ width*randn(N,1);
ytrain=(Ny/2)+ width*randn(N,1);
Ztrain=[xtrain,ytrain];
% Evaluate the density in grid for visualization
xim=repmat((1:Nx)',1,Ny);
yim=repmat((1:Ny),Nx,1);
xtest=reshape(xim,Nx*Ny,1);
ytest=reshape(yim,Nx*Ny,1);
Ztest=[xtest,ytest];
% validation set for tuning of h
% Normal distributed data for training set
xval=(Nx/2)+ width*randn(Nval,1);
yval=(Ny/2)+ width*randn(Nval,1);
Zval=[xval,yval];
% tune h between limits
Nh=20;
hmax=100;
hmin=1;
harray=linspace(hmin,hmax,Nh);
for n=1:Nh,
    disp(['Evaluating h ',int2str(n),' of ',int2str(Nh)])
    h=harray(n);
    [estpdf]=kpdf(Ztrain,h,Zval);
    Error(n)=sum(-log(estpdf))/Nval;
end,
figure(1)
plot(harray,Error,'o',harray,Error,'-')
xlabel('h'), ylabel('Validation error'),grid
[dummy, ih]=sort(Error);
h_opt=harray(ih(1));
%%%%%%%%% Now plot h= h_opt, h = 0.1*h_opt, h=100*h_opt
figure(2)
estpdf=kpdf(Ztrain,h_opt,Ztest);
subplot(1,2,1),imagesc(reshape(estpdf,Nx,Ny))
subplot(1,2,2),mesh(reshape(estpdf,Nx,Ny))
axis([0 Nx 0 Ny 0 1.1*max(estpdf)])
figure(3)
estpdf=kpdf(Ztrain,0.1*h_opt,Ztest);
subplot(1,2,1),imagesc(reshape(estpdf,Nx,Ny))
subplot(1,2,2),mesh(reshape(estpdf,Nx,Ny))
axis([0 Nx 0 Ny 0 1.1*max(estpdf)])
figure(4)
estpdf=kpdf(Ztrain,100*h_opt,Ztest);
subplot(1,2,1),imagesc(reshape(estpdf,Nx,Ny))
subplot(1,2,2),mesh(reshape(estpdf,Nx,Ny))
axis([0 Nx 0 Ny 0 1.1*max(estpdf)])