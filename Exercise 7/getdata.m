function [xtrain,xtest]=getdata(Ntrain,Ntest,noise)
%function [xtrain,xtest]=getdata(Ntrain,Ntest,noise)
%
% Creates 2D data with 3 clusters of width noise
% 
% (c) Lars Kai Hansenn
%
x11=0.75+noise*randn(round(round(Ntrain/3)),1);
x12=0.25+noise*randn(round(round(Ntrain/3)),1);
x13=0.5+noise*randn(round(round(Ntrain/3)),1);
x21=0.25+noise*randn(round(round(Ntrain/3)),1);
x22=0.25+noise*randn(round(round(Ntrain/3)),1);
x23=0.5+noise*randn(round(round(Ntrain/3)),1);
x1=[x11; x12; x13];
x2=[x21; x22; x23];
xtrain=[x1,x2];


z11=0.75+noise*randn(round(Ntest/3),1);
z12=0.25+noise*randn(round(Ntest/3),1);
z13=0.5+noise*randn(round(Ntest/3),1);
z21=0.25+noise*randn(round(Ntest/3),1);
z22=0.25+noise*randn(round(Ntest/3),1);
z23=0.5+noise*randn(round(Ntest/3),1);
z1=[z11; z12; z13];
z2=[z21; z22; z23];
xtest=[z1,z2];



