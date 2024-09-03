function [output,Y] = appKernelLogisticRegression(model,X,XTrain)
G = getKernelRBF(X,XTrain,4);
Y = logsig(model.W * G + model.B);
output = reshape(Y,1,[]);
output = [1-output;output];
end