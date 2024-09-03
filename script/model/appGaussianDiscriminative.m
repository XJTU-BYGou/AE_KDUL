function [output,Y] = appGaussianDiscriminative(Model,X)

delta = 1e-8;
numFeature = size(Model.Mu,2);
numClasses = size(Model.Mu,1);
invtau = zeros(size(Model.Sigma));
dettau = zeros(numClasses,1);
for j = 1:numClasses
    invtau(:,:,j) = inv(Model.Sigma(:,:,j));
    dettau(j,1) = det(Model.Sigma(:,:,j)) + 0;
end

dim = numFeature;
lh = [diag(Model.Alpha(1).*exp(-0.5.*...
        (X-Model.Mu(1,:))*invtau(:,:,1)*(X- Model.Mu(1,:))')...
        ./sqrt((2*pi)^dim*dettau(1)))';
        diag(Model.Alpha(2).*exp(-0.5.*...
        (X-Model.Mu(2,:))*invtau(:,:,2)*(X- Model.Mu(2,:))')...
        ./sqrt((2*pi)^dim*dettau(2)))'];

Y = lh;
lh = lh + delta;
output = lh./sum(lh);
end