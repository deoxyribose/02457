function j=getint(q)
%function j=getint(q)
% get random integer from distribution
%  with cumulative distribution q
%
r=rand;
j=min(find(r<q));