function [n,x,data] = hist2d(data,b,range,nm)
% Calculate or draw a 2D histogram of a data matrix.
%
% [n,x] = hist2d(DATA,B,RANGE,NORMALIZE)
%
% where
%   DATA:  2xM matrix. M samples of a 2D quantity.
%   B:     The number of bins in each dimension, i.e the data will 
%          be divided on a BxB mesh grid (default value = 10).
%   RANGE: range of the mesh grid = [xmin xmax ymin ymax]. 
%          Default value is the range of the data. RANGE = [] also
%          corresponds to the range of the data.
%   NORMALIZE: 1 if the histogram should be normalized (corresponds
%          to a PDF. 0 if histogram should not be normalized. 
%          Default value is 0.
%
% hist2d(...) without output arguments produces a 3D histogram bar plot of
%   the results.
%

% (c) Karam Sidaros, August 1999. 
% vectorized by Karam and Torben July 2001

%%%%%%%%%%%%%%%%%%%%%%%%% Checks and Defaults %%%%%%%%%%%%%%%%%%%%
[d1 d2] = size(data);
if d1 ~= 2
  if d2 == 2
    data = data';
  else
    error('DATA must be a 2xM matrix');
  end
end
ndata = size(data,2);  % number of datasets

if nargin < 2
  b = 10;
end

if nargin < 3
  fullrange = 1;
elseif  isempty(range)
  fullrange = 1;
else
  fullrange = 0;
end

if fullrange  
  dmin = min(data,[],2)';
  dmax = max(data,[],2)';
  range = [dmin-10*eps*abs(dmin);dmax+10*eps*abs(dmax)];  %ensures edges are included
  range = [range(:)]';
else 
  dmin=range([1 3]);
end

if nargin < 4
  nm = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%% classification %%%%%%%%%%%%%%%%%%%%%%%%%%%

dx1 = (range(2)-range(1))/b;
dx2 = (range(4)-range(3))/b;

data(2,:)=data(2,:)-dmin(2);
data(2,:)=dx1*b*(floor(data(2,:)/dx2));

data=sum(data,1);


x= (dmin(1)+(dx1/2)):dx1:b*b*dx1+dmin(1);

[n,x]=hist(data,x);

%keyboard
n=reshape(n,[b b])';


clear x
x(1,:) = range(1)+(dx1/2):dx1:range(2)-(dx1/2);
x(2,:) = range(3)+(dx2/2):dx2:range(4)-(dx2/2);




%%%%%%%%%%%%%%%%%%%%%%%% Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%

if nm
  N = sum(n(:));
  ar=(x(1,2)-x(1,1))*(x(2,2)-x(2,1));  % area of each "pixel"
  n = n/N/ar;   % Normalization of histogram
end

