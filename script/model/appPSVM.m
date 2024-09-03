function [output,Y] = appPSVM(model,X)

Y = logsig(model.weights.W * X' + model.weights.B);
Y = logsig(Y.*model.weights.A(1,1)+model.weights.A(2,1));
output = reshape(Y,1,[]);
output = [1-output;output];
end