function [model,output,history,options] = trainLogisticRegression_rand(model,...
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
initialLearnRate = 1e-3;
gradientDecayFactor = 0.9;
learnRateDropPeriod = 1e4;
learnRateDropFactor = 0.8;
Verbose = true;
VerboseFrequency = 50;

options.InitialLearnRate = initialLearnRate;
options.GradientDecayFactor = gradientDecayFactor;
options.LearnRateDropPeriod = learnRateDropPeriod;
options.LearnRateDropFactor = learnRateDropFactor;
options.Verbose = Verbose;
options.VerboseFrequency = VerboseFrequency;
options.MaxEpoch = MaxEpoch;

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
    
W = dlarray(randn(numClasses,numFeature));
B = dlarray(randn(numClasses,1));

Model.W = W;
Model.B = B;
else
Model = model.LR.Model;
end
vel_W = [];
vel_B =[];
averageGrad_W = [];
averageSqGrad_W = [];
averageGrad_B = [];
averageSqGrad_B = [];


tic;
for i = 1:options.MaxEpoch
    % Calculate custom Loss and gradient

    [output,loss,gradloss] = dlfeval(@lossOfModel_rand,Model,XTrain,lossfun);
    output = extractdata(output);
    loss = extractdata(loss);
    % Update parameters
    lr = initialLearnRate;

    [Model.W,averageGrad_W,averageSqGrad_W] = adamupdate(Model.W,gradloss.W,averageGrad_W,averageSqGrad_W,i,lr);
    [Model.B,averageGrad_B,averageSqGrad_B] = adamupdate(Model.B,gradloss.B,averageGrad_B,averageSqGrad_B,i,lr);
    
    
    model.LR.Model = Model;
    model.LR.Loss = loss;
    if loss < model.LR.BestModel.Loss
        model.LR.BestModel.Loss = loss;
        model.LR.BestModel.Model = Model;
    end
    
    
    %% Record the processing
    
    curLoop.Acc = mean(double(YTrain')-1 == (output(end,:)>0.5));
    curLoop.Loss = double(loss);
    curLoop.ElapsedTime = toc;
    history(i,1) = curLoop;
    if options.Verbose
        if mod(i,VerboseFrequency) == 0 || i == 1
            fprintf('Epoch: %i , Training Time: %f , Loss: %f \n',...
                i,curLoop.ElapsedTime,curLoop.Loss);
        end
        % Plot
%         if i == 1
%             fig = figure('Position',[200,150,680,800]);
%             axbg = axes(fig,'Units','pixels','Position',[100 460 500 300],...
%             'Color', 'none','Box','off',...
%             'XAxisLocation','top','YAxisLocation','right',...
%             'LineWidth',2,'TickLength', [0.02,0.05],...
%             'XTick',[],'YTick',[]);
%             ax1 = axes(fig,'Units','pixels','Position',axbg.Position,...
%             'Color', 'none','Box','off',...
%             'LineWidth',2,'TickLength', [0.02,0.05],...
%             'FontName','Arial','FontSize',16,'FontWeight','bold');    
%             xlabel('Iteration');
%             ylabel('Pseudo Accuracy (%)');
%             ax1.YLim = [0,100];
%             hold on;
%             anAcc = animatedline(ax1,i,curLoop.Acc.*100,'Color','b','LineWidth',2);
%         
%             axbg = axes(fig,'Units','pixels','Position',[100 80 500 300],...
%             'Color', 'none','Box','off',...
%             'XAxisLocation','top','YAxisLocation','right',...
%             'LineWidth',2,'TickLength', [0.02,0.05],...
%             'XTick',[],'YTick',[]);
%             ax2 = axes(fig,'Units','pixels','Position',axbg.Position,...
%             'Color', 'none','Box','off',...
%             'LineWidth',2,'TickLength', [0.02,0.05],...
%             'FontName','Arial','FontSize',16,'FontWeight','bold');    
%             hold on;
%             xlabel('Iteration');
%             ylabel('Loss');
%             anLoss = animatedline(ax2,i,curLoop.Loss,'Color','r','LineWidth',2);
%             drawnow();
%         else
%             addpoints(anAcc,i,curLoop.Acc.*100);
%             addpoints(anLoss,i,curLoop.Loss);
%             drawnow();
%         end
    end
end
end

function [output,loss,gradloss] = lossOfModel_rand(model,X,lossfun)
        
        Y = logsig(model.W * X' + model.B);
        lambda = 5e-4;
        
        output = reshape(Y,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;

        gradloss.W = dlgradient(loss,model.W);
        gradloss.B = dlgradient(loss,model.B);

end
