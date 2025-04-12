function [output,Y] = appLogisticRegression(model,X)
switch size(model.W,1)
    case 1
    Y = logsig(model.W * X' + model.B);
    output = reshape(Y,1,[]);
    output = [1-output;output];
    otherwise
    Y = exp(model.W * X' + model.B);
    output = Y./sum(Y,1);
end
end