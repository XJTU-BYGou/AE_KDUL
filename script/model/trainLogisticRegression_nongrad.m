function [model,output,history,options] = trainLogisticRegression_nongrad(model,...
    XTrain,YTrain,lossfun,varargin)

rng(0);
% Check the input arguments
if nargin < 4
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end

% Get Properties
MaxEpoch = 10e3;
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
model.LR = [];
model.LR.Model = [];
model.LR.BestModel.Loss = inf;
model.LR.BestModel.Model = [];
    
W = (randn(numClasses,numFeature));
B = (randn(numClasses,1));

Model.W = W;
Model.B = B;
else
Model = model.LR.Model;
end



tic;
[estParam,exitflag,optProcess] = ga(@(x)lossOfModel_nongrad(x,XTrain,lossfun),numFeature+1,[],[],[],[],[],[],[],options);
Model.W = estParam(1:end-1);
Model.B = estParam(end);

model.LR.Model = Model;
[loss,output] = lossOfModel([Model.W,Model.B],XTrain,lossfun);
model.LR.Loss = loss;
model.LR.BestModel.Loss = loss;
model.LR.BestModel.Model = Model;

curLoop.Acc = mean(double(YTrain')-1 == (output(end,:)>0.5));
curLoop.Loss = double(loss);
curLoop.ElapsedTime = toc;
curLoop.OptProcess = optProcess;
curLoop.Finish = exitflag;
history = curLoop;

% for i = 1:options.MaxEpoch
%     % Update parameters   
%     
%     
%     % Calculate custom Loss and gradient
%     [output,loss] = dlfeval(@lossOfModel_rand,Model,XTrain,lossfun);
%     
%     model.LR.Model = Model;
%     model.LR.Loss = loss;
%     if loss < model.LR.BestModel.Loss
%         model.LR.BestModel.Loss = loss;
%         model.LR.BestModel.Model = Model;
%     end
%     
%     %% Record the processing
%     
%     curLoop.Acc = mean(double(YTrain')-1 == (output(end,:)>0.5));
%     curLoop.Loss = double(loss);
%     curLoop.ElapsedTime = toc;
%     history(i,1) = curLoop;
%     if options.Verbose
%         if mod(i,VerboseFrequency) == 0 || i == 1
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

function [output,loss] = lossOfModel_rand(model,X,lossfun)
        
        Y = logsig(model.W * X' + model.B);
        lambda = 5e-4;
        
        output = reshape(Y,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;

end

function [loss] = lossOfModel_nongrad(param,X,lossfun)
        Y = logsig(param(1:end-1) * X' + param(end));
        lambda = 5e-4;
        
        output = reshape(Y,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;
end
function [loss,output] = lossOfModel(param,X,lossfun)
        
        Y = logsig(param(1:end-1) * X' + param(end));
        lambda = 5e-4;
        
        output = reshape(Y,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;

end