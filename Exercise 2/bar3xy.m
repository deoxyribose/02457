function h = bar3xy(x,y,z,varargin)
% 3D bar plot with values on both X and Y axes.
%
% bar3xy(X,Y,Z,...) gives the same plot as bar3(Z,...) except
%   that the values in the vectors X and Y are plotted on the
%   x- and y-axes.
%   If Z is an MxN matrix, X must be of length N (columns) and 
%   Y must be of length M (rows).
%
%   X and Y must be monotonically increasing and equally spaced.
%
% H = bar3xy(...) returns a vector of surface handles.

%
% (c) Karam Sidaros, August 1999.
%

%%%%%%%%%%%%%%%%%%%%%%%% Check inputs %%%%%%%%%%%%%%%%%%%%%%%

if nargin <3
  error('Not enough arguments');
end

[m n] = size(z);
sx = size(x);
sy = size(y);

if min(sx) > 1  
  error('X must be a vector');
end  

if min(sy) > 1  
  error('Y must be a vector');
end  

if sx(1) == 1
  x = x';
end
if sy(1) == 1
  y = y';
end

if prod(sx) ~= n | prod(sy) ~= m
  error('X and Y lengths do not match Z');
end

%%%%%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%

h=bar3(y,z,varargin{:});
nticks = length(get(gca,'ytick'));
dx = x(2)-x(1);
%%%% tranformation from one scale to X scale
a1 = dx;
b1 = x(1)-dx;
for a = 1:length(h)
  set(h(a),'xdata', get(h(a),'xdata')*a1+b1);
end

set(gca,'xtickmode','auto','xlim',[x(1)-dx/2 x(n)+dx/2]);




