function [output,Y] = appKernelSupportVectorMachine(model,X,XTrain)
G = getKernelRBF(X,XTrain,4);
Y = model.weights.Alpha .* model.Y * G + model.weights.B;
Y = logsig(Y.*model.weights.A(1,1)+model.weights.A(2,1));
output = reshape(Y,1,[]);
output = [1-output;output];
end