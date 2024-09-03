function G = getKernelRBF(x1,x2,sigma)
% x1,x2  -  n * d Matrix; n - number of samples, d - number of features
if nargin < 2
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
elseif nargin == 2
    sigma = 1;
end


G = zeros(size(x1,1),size(x2,1));
for i = 1:size(x1,1)
    G(i,:) = exp(-sum((x1(i,:) - x2).^2,2)/(sigma^2))';
end

end