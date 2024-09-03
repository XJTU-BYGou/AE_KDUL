function [output,Y] = appGBDT(model,X)

Y = zeros(size(model,2),size(X,1));
for i = 1:size(model,1)
    Y = Y + [predict(model{i,1},X)';predict(model{i,2},X)'];
end
output = exp(Y)./sum(exp(Y));
    
end