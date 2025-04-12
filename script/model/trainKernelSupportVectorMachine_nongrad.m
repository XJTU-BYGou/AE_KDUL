function [model,output,history,options] = trainKernelSupportVectorMachine_nongrad(model,...
    XTrain,YTrain,lossfun,varargin)
% Update parameters with Pegasos algorithm.
% Pegasos:  primal estimated sub-gradient solver for SVM

rng(0);


% Check the input arguments
if nargin < 3
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end

G = getKernelRBF(XTrain,XTrain,4);

% Get Properties
MaxEpoch = 10e3;
% MaxEpoch = 20e3;
[numSample,numFeature] = size(XTrain);
numClasses = 1;
% history = struct;


% Set Parameters

options = optimoptions('ga');

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'options'
        options = varargin{i*2};
    end
end

% Init model
if isempty(model)
model.KSVM = [];
model.KSVM.Model = [];
model.KSVM.BestModel.Loss = inf;
model.KSVM.BestModel.Model = [];
    
W = (zeros(numClasses,numSample));
B = (zeros(numClasses,1));
Alpha = (zeros(numClasses,numSample));
A = ([randn(numClasses,1);0]);

Model.weights.B = B;
Model.weights.Alpha = Alpha;
Model.weights.A = A;

Model.Y = zeros(size(Model.weights.Alpha));
else
Model = model.KSVM.Model;
end

tic;
numParam = numel(Model.weights.Alpha) + numel(Model.weights.B) + numel(Model.weights.A)...
    + numel(Model.Y);
[estParam,exitflag,optProcess] = ga(@(x)lossOfModel_nongrad(x,Model,G,lossfun),numParam,[],[],[],[],[],[],[],options);
Model = vec2mdl(estParam,Model);

model.KSVM.Model = Model;
[loss,output] = lossOfModel(Model,G,lossfun);
model.KSVM.Loss = loss;

model.KSVM.BestModel.Loss = loss;
model.KSVM.BestModel.Model = Model;
curLoop.Acc = mean(double(YTrain')-1 == (output(end,:)>0.5));
curLoop.Loss = loss;
curLoop.ElapsedTime = toc;
curLoop.OptProcess = optProcess;
curLoop.Finish = exitflag;
history = curLoop;

% 
% for i = 1:options.MaxEpoch
%    
%     Y = Model.weights.Alpha .* Model.Y * G + Model.weights.B;
%     
%     YL = logsig(Y.*Model.weights.A(1,1)+Model.weights.A(2,1));
%     % Calculate custom Loss and gradient
%     [output,loss,gradloss] = dlfeval(@lossOfModel_rand,Model,YL,lossfun);
%     output = extractdata(output);
%     loss = extractdata(loss);
%     Y = extractdata(Y);
% 
%     % Update parameters
%     lr = initialLearnRate;
%     
%     YL_pre = extractdata(sign(-gradloss.X));
%     
% 
%     sv = (YL_pre .* (Y.*Model.weights.A(1,1)+Model.weights.A(2,1))) <= 0;
%     batchFlag = rand(size(Model.weights.Alpha)) < 1;
%     grad.weights.Alpha = 0.*Model.weights.Alpha - batchFlag.* (sv);
%     grad.weights.B = - (batchFlag.* sv) * YL_pre'./(sum(sv)+1e-6);
%     
%     grad.weights.A(1,1) = sum(batchFlag.* gradloss.X.*YL.*(1-YL).*Y);
%     grad.weights.A(2,1) = sum(batchFlag.* gradloss.X.*YL.*(1-YL));
%     
%     [Model.weights,averageGrad,averageSqGrad] = adamupdate(Model.weights,grad.weights,averageGrad,averageSqGrad,i,lr);
%     
%     Y = Model.weights.Alpha .* YL_pre * G + Model.weights.B;
%     YL = logsig(Y.*Model.weights.A(1,1)+Model.weights.A(2,1));
%     Model.Y = sign(YL-0.5);
%     
%     
%     
%     model.KSVM.Model = Model;
%     model.KSVM.Loss = loss;
%     if loss < model.KSVM.BestModel.Loss
%         model.KSVM.BestModel.Loss = loss;
%         model.KSVM.BestModel.Model = Model;
%     end
% 
%     %% Record the processing
%     
%     curLoop.Acc = mean(double(YTrain')-1 == (output(end,:)>0.5));
%     curLoop.Loss = loss;
%     curLoop.ElapsedTime = toc;
%     history(i,1) = curLoop;
%     if options.Verbose
%         if mod(i,options.VerboseFrequency) == 0 || i == 1
%             fprintf('Epoch: %i , Training Time: %f , Loss: %f \n',...
%                 i,curLoop.ElapsedTime,curLoop.Loss);
%         end
%         % Plot
% %         if i == 1
% %             fig = figure('Position',[200,150,680,800]);
% %             axbg = axes(fig,'Units','pixels','Position',[100 460 500 300],...
% %             'Color', 'none','Box','off',...
% %             'XAxisLocation','top','YAxisLocation','right',...
% %             'LineWidth',2,'TickLength', [0.02,0.05],...
% %             'XTick',[],'YTick',[]);
% %             ax1 = axes(fig,'Units','pixels','Position',axbg.Position,...
% %             'Color', 'none','Box','off',...
% %             'LineWidth',2,'TickLength', [0.02,0.05],...
% %             'FontName','Arial','FontSize',16,'FontWeight','bold');    
% %             xlabel('Iteration');
% %             ylabel('Pseudo Accuracy (%)');
% %             ax1.YLim = [0,100];
% %             hold on;
% %             anAcc = animatedline(ax1,i,curLoop.Acc.*100,'Color','b','LineWidth',2);
% %         
% %             axbg = axes(fig,'Units','pixels','Position',[100 80 500 300],...
% %             'Color', 'none','Box','off',...
% %             'XAxisLocation','top','YAxisLocation','right',...
% %             'LineWidth',2,'TickLength', [0.02,0.05],...
% %             'XTick',[],'YTick',[]);
% %             ax2 = axes(fig,'Units','pixels','Position',axbg.Position,...
% %             'Color', 'none','Box','off',...
% %             'LineWidth',2,'TickLength', [0.02,0.05],...
% %             'FontName','Arial','FontSize',16,'FontWeight','bold');    
% %             hold on;
% %             xlabel('Iteration');
% %             ylabel('Loss');
% %             anLoss = animatedline(ax2,i,curLoop.Loss,'Color','r','LineWidth',2);
% %             drawnow();
% %         else
% %             addpoints(anAcc,i,curLoop.Acc.*100);
% %             addpoints(anLoss,i,curLoop.Loss);
% %             drawnow();
% %         end
%     end
% end
end


function [loss] = lossOfModel_nongrad(param,Model,XTrain,lossfun)
        Model = vec2mdl(param,Model);
        Y = svmCalcu(Model,XTrain);
        
        output = reshape(Y,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;
end
function [loss,output] = lossOfModel(Model,XTrain,lossfun)
        
        Y = svmCalcu(Model,XTrain);
        
        output = reshape(Y,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;
end

function param = mdl2vec(Model)
param = [Model.weights.Alpha(:);Model.weights.B(:);Model.weights.A(:);...
    Model.Y(:)];
end

function Model = vec2mdl(param,Model)
[numClasses,numFeature] = size(Model.weights.Alpha);

Model.weights.Alpha = reshape(param(1:numClasses*numFeature),size(Model.weights.Alpha));
Model.weights.B = reshape(param(numClasses*numFeature+1:numClasses+numFeature*numClasses),size(Model.weights.B));
Model.weights.A = reshape(param(numClasses+numFeature*numClasses+1:numClasses+numFeature*numClasses+numel(Model.weights.A)),size(Model.weights.A));
Model.Y = sign(reshape(param(numClasses+numFeature*numClasses+numel(Model.weights.A)+1:end),size(Model.Y)));
end

function YL = svmCalcu(Model,G)
    Y = Model.weights.Alpha .* Model.Y * G + Model.weights.B;
    YL = logsig(Y.*Model.weights.A(1,1)+Model.weights.A(2,1));
end