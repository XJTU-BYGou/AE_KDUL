function [res,record] = AEMixExponent(E,label)
% Estimiate the performance of partition of AE sample from Energy Exponent
% of Power law.
% Author: Gby, Power by MATLAB 2021a
% 
% Update:
%     

E = reshape(E,1,[]);
if nargin < 1 || nargin > 3
    error('Too many or too few input arguments.');
elseif nargin == 1
    label = ones(size(E));
end

if isempty(label)
    label = ones(size(E));
end

if length(E) ~= length(label)
    error('Different Size Error.')
end


labelCont = unique(label);
if isnumeric(labelCont)
    labelCont(labelCont==0) = [];
end

E_eff = cell(length(labelCont),1);
for i = 1:length(labelCont)
    tmpE = E(label==labelCont(i));
    % MLE
    [res(i,1),record(i,1)] = powerlawExponentMLEstimator(tmpE,[]);
    
end
    




end