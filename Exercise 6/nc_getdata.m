function [tr_i,tr_t,te_i,te_t] = nc_getdata

%NC_GETDATA    Neural classifier get forensic glass data
%  [tr_i,tr_t,te_i,te_t] = nc_getdata
%  Use an example data set: the forensic glass data from the Proben
%  collection as data 
%  
%  Neural classifier, DSP IMM DTU, MWP97

%  cvs $Revision: 1.1 $

  load glass1.dat

  tr_i = glass1(1:107,1:9);
  tr_tar = glass1(1:107,10:15);
  [v p] = max(tr_tar');
  tr_t = p';
  
  te_i = glass1(108:160,1:9);
  te_tar = glass1(108:160,10:15);
  [v p] = max(te_tar');
  te_t = p';
  
