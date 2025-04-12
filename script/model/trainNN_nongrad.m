function [model,output,history,options] = trainNN_nongrad(model,...
    XTrain,YTrain,lossfun,varargin)
rng(0);
% Check the input arguments
if nargin < 3
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end
delta = 1e-10;
MaxEpoch = 1e4;
% Get Properties
[numSample,numFeature] = size(XTrain);
numClasses = 2;

% Set Parameters
options = optimoptions('ga');

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'options'
        options = varargin{i*2};
    end
end


fcLayerInd = find(arrayfun(@(x)(isprop(x,'Weights') & isprop(x,'Bias')),model));
numParam = 0;
lastInputSize = size(XTrain,2);
for i = 1:numel(fcLayerInd)
numParam = numParam + (lastInputSize+1)*model(fcLayerInd(i)).OutputSize;
end

tic;
[estParam,exitflag,optProcess] = ga(@(x)lossOfModel_nongrad(x,model,XTrain,lossfun),numParam,[],[],[],[],[],[],[],options);
%%
Model = vec2mdl(estParam,model);

model = Model;
[loss,output] = lossOfModel(Model,XTrain,lossfun);

[~,YL] = max(output);
curLoop.Acc = mean(double(YTrain') == YL);
curLoop.Loss = double(loss);
curLoop.ElapsedTime = toc;
curLoop.OptProcess = optProcess;
curLoop.Finish = exitflag;
history = curLoop;
end

function [loss] = lossOfModel_nongrad(param,layer,XTrain,lossfun)
        Model = vec2mdl(param,layer);
        Y = NNCalcu(Model,XTrain);
        
        output = Y;
        L = lossfun(output);
        
        loss =  L;
end
function [loss,output] = lossOfModel(layer,XTrain,lossfun)
        Y = NNCalcu(layer,XTrain);
        
        output = Y;
        L = lossfun(output);
        
        loss =  L;
end


function Model = vec2mdl(param,layer)
fcLayerInd = find(arrayfun(@(x)(isprop(x,'Weights') & isprop(x,'Bias')),layer));
lastInputSize = layer(1).InputSize;
curInd = 0;
% for i = 1:numel(fcLayerInd)
% Model(i).Weights = reshape(param(curInd+1:curInd+lastInputSize*layer(fcLayerInd(i)).OutputSize),layer(fcLayerInd(i)).OutputSize,lastInputSize);
% Model(i).Bias = reshape(param(curInd+lastInputSize*layer(fcLayerInd(i)).OutputSize+1:curInd+(lastInputSize+1)*layer(fcLayerInd(i)).OutputSize),layer(fcLayerInd(i)).OutputSize,1);
% lastInputSize = layer(fcLayerInd(i)).OutputSize;
% end
Model = layer;
for i = 1:numel(fcLayerInd)
Model(fcLayerInd(i)).Weights = reshape(param(curInd+1:curInd+lastInputSize*layer(fcLayerInd(i)).OutputSize),layer(fcLayerInd(i)).OutputSize,lastInputSize);
Model(fcLayerInd(i)).Bias = reshape(param(curInd+lastInputSize*layer(fcLayerInd(i)).OutputSize+1:curInd+(lastInputSize+1)*layer(fcLayerInd(i)).OutputSize),layer(fcLayerInd(i)).OutputSize,1);
lastInputSize = layer(fcLayerInd(i)).OutputSize;
end
end

function y = NNCalcu(Model,XTrain)
% y = XTrain;
% for i = 1:numel(Model)
%     y = y*model(i).Weights+Model(i).Bias;
%     y(y<0) = 0.01.* y(y<0);
% end
lgraph = layerGraph(Model);
y = extractdata(forward(dlnetwork(lgraph),dlarray(XTrain,'BC')));
end