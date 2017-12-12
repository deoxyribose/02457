function h = arrow(x,y,w,wa,p,c);
%arrow(x,y,width,width_arrow,p,colour) 
%  draws an arrow between points in x and y.
%  width is the width of the line
%  width_arrow is the relative width of the arrow head
%    compared to the width of the line
%  p is the relative length of the arrow head compared 
%    to the total length
%  colour is the colour of the arrow.
%  
%  h = arrow(...) returns the handle for the arrow.
%

warning off

if nargin < 6
  c = 'k';
end
if nargin < 5
  p = 0.05;
end


dx = x(2)-x(1);
dy = y(2)-y(1);

theta = atan(dy/dx)-pi/2;

if (theta<0) & (dx<0)
  theta = theta + pi;
end  

l = sqrt(dx^2 + dy^2);

x0 = [0.5 0.5 0.5*(wa) 0 -0.5*(wa) -0.5 -0.5]*w;
y0 = [0   1-p 1-p        1  1-p        1-p  0  ]*l;

x1 = x(1)+real((x0 + i*y0) * exp(i*theta));
y1 = y(1)+imag((x0 + i*y0) * exp(i*theta));

h= patch(x1,y1,c);
warning on

