function [model,output,history,options] = trainGBDT_rand(model,...
    XTrain,YTrain,lossfun,varargin)
rng(0);
% Check the input arguments
if nargin < 3
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end


% Get Properties
MaxEpoch = 1e3;
% MaxEpoch = 20e3;
[numSample,numFeature] = size(XTrain);
numClasses = 2;
% history = struct;


% Set Parameters
initialLearnRate = 1e1;
gradientDecayFactor = 0.9;
learnRateDropPeriod = 1e4;
learnRateDropFactor = 0.8;
Verbose = true;
VerboseFrequency = 10;

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
model.GBDT = [];
model.GBDT.Model = [];
model.GBDT.BestModel.Loss = inf;
model.GBDT.BestModel.Model = [];
model.GBDT.TrainingY = [];
f = zeros(numClasses,numSample);
else
f = model.GBDT.TrainingY;
end



tic;

residual = zeros(numSample,numClasses);
for i = 1:options.MaxEpoch


    tree1 = fitrtree(XTrain,residual(:,1),'MinLeafSize',5);
    tree2 = fitrtree(XTrain,residual(:,2),'MinLeafSize',5);
    
    model.GBDT.Model = [model.GBDT.Model;[{tree1},{tree2}]];    
    
    Y = f + [predict(tree1,XTrain)';predict(tree2,XTrain)'];
    
    % Calculate custom Loss and gradient
    [output,loss,gradloss] = dlfeval(@lossOfModel_rand,model,dlarray(Y),lossfun);
    output = extractdata(output);
    loss = extractdata(loss);

    % Update parameters
    lr = options.InitialLearnRate;
    
    residual = - lr .* extractdata(gradloss.X');
    f = Y;
    
    model.GBDT.Loss = loss;
    model.GBDT.TrainingY = f;
    if loss < model.GBDT.BestModel.Loss
        model.GBDT.BestModel.Loss = loss;
        model.GBDT.BestModel.Index = size(model.GBDT.Model,1);
    end

    %% Record the processing
    [~,YL] = max(f);
    curLoop.Acc = mean(double(YTrain') == YL);
    curLoop.Loss = loss;
    curLoop.ElapsedTime = toc;
    history(i,1) = curLoop;
    if options.Verbose
        if mod(i,options.VerboseFrequency) == 0 || i == 1
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

        output = exp(X)./sum(exp(X));
        L = lossfun(output);
        
        loss =  L;
        gradloss.X = dlgradient(loss,X);

end

