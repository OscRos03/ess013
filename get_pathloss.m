function [A,y] = get_pathloss(d,d0,l_d0,a,sigma2)
%{
Function to generate measurements y, of the pathloss at distances d, given
- vector d, of distances
- scalar d0, reference distance at which we have known pathloss
- scalar l_d0, known pathloss at distance d0
- scalar a, pathloss exponent
- scalar sigma2, variance of noise in s-term
%}

% Get length of vector d
n = length(d);
% Construct A matrix
A = [ones(n,1), 10*log10(d./d0)];
% Generate noise samples
s = randn([n,1])*sqrt(sigma2);
% Construct x
x = [l_d0; a];
% Calculate y
y = A*x+s;

end