function [xarray,p,xhistarray,phist]=norm1d(mu,sigma2,xmin,xmax,N,dx)
% function [p,phist]=norm1d(xmin,xmax,dx)
%
% Function computes the density of the 1D normal distribution
% at N points in the interval [xmin,xmax]
% and a histogram with bin-width dx in the same interval.
%
% INPUT
%
%  mu       mean value
%  sigma2   variance
%  xmin     lower interval limit
%  xmax     upper interval limit
%  N        number of points in the interval
%  dx       bin width of histogram in the interval [xmin,xmax].
%
% OUTPUT
%
% p         density values N*1 array
% phist     histogram values  ceil((xmax-xmin)/dx) * 1 array

xarray=xmin + ((1:N)' - 1)*(xmax-xmin)/(N-1);   %range of x values for the density
Nhist= ceil((xmax-xmin)/dx);                 % numer of points in the histogram
xhistarray=xmin + ((1:Nhist)' - 1)*(xmax-xmin)/(Nhist-1);   %range of x values for the histogram
normconst=(1/sqrt(2*pi*sigma2));
p=normconst*exp(-(1/2/sigma2)*(xarray-mu).^2);
phist=normconst*dx*exp(-(1/2/sigma2)*(xhistarray-mu).^2);

