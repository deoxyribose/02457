function cl = clevels(cm)
% finds the contour levels in the contour matrix cm
% 
% cl = clevels(cm)
%   gives a vector, cl, with the contours

% (c) Karam Sidaros, August, 1999.
%

a=1;
  cl = [0];
  l = size(cm,2);
  while a <= l
    cl = [cl cm(1,a)];
    a = a+cm(2,a)+1;    
  end
  