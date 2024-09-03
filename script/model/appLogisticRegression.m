function [output,Y] = appLogisticRegression(model,X)

Y = logsig(model.W * X' + model.B);
output = reshape(Y,1,[]);
output = [1-output;output];
end