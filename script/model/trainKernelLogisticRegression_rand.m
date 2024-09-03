function [model,output,history,options] = trainKernelLogisticRegression_rand(model,...
    XTrain,YTrain,lossfun,varargin)
% Check the input arguments
if nargin < 4
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end

% Get Properties
% MaxEpoch = 2e3;
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
options.Optimizer = 'Adam'; %'SGD';

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'options'
        options = varargin{i*2};
    end
end


% Init model
if isempty(model)
model.KLR = [];
model.KLR.Model = [];
model.KLR.BestModel.Loss = inf;
model.KLR.BestModel.Model = [];
    
A = dlarray(zeros(numClasses,numSample));
W = dlarray(zeros(numClasses,numSample));
B = dlarray(zeros(numClasses,1));

Model.A = A;
Model.W = W;
Model.B = B;
else
Model = model.KLR.Model;
    
end
vel_W = [];
vel_B =[];
averageGrad_W = [];
averageSqGrad_W = [];
averageGrad_B = [];
averageSqGrad_B = [];

tic;
G = getKernelRBF(XTrain,XTrain,4);
for i = 1:options.MaxEpoch
    % Update parameters
    lr = initialLearnRate ;
    
    switch options.Optimizer
        case 'SGD'
    Model.W = Model.A;
    Y = logsig(Model.W * G + Model.B);
    
    % Calculate custom Loss and gradient
    [output,loss,Model,gradloss] = dlfeval(@lossOfModel_rand,Model,Y,lossfun);
    output = extractdata(output);
    loss = extractdata(loss);

    % Update parameters
    Model.A = Model.A - lr * extractdata(gradloss.X);
    Model.B = Model.B - lr * mean(extractdata(gradloss.X),2);
        case 'Adam'
    
    % Calculate custom Loss and gradient
    [output,loss,Model,gradloss] = dlfeval(@lossOfModel_adam_rand,Model,G,lossfun);
    output = extractdata(output);
    loss = extractdata(loss);

    % Update parameters
    [Model.W,averageGrad_W,averageSqGrad_W] = adamupdate(Model.W,gradloss.W,averageGrad_W,averageSqGrad_W,i,lr);
    [Model.B,averageGrad_B,averageSqGrad_B] = adamupdate(Model.B,gradloss.B,averageGrad_B,averageSqGrad_B,i,lr);
    
    end
    
    model.KLR.Model = Model;
    model.KLR.Loss = loss;
    if loss < model.KLR.BestModel.Loss
        model.KLR.BestModel.Loss = loss;
        model.KLR.BestModel.Model = Model;
    end
    
    
    %% Record the processing
    [~,YL] = max(output);
    curLoop.Acc = mean(double(YTrain) == YL');
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

function [output,loss,model,gradloss] = lossOfModel_rand(model,X,lossfun)
        
        output = reshape(X,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;

        gradloss.X = dlgradient(loss,X);
        
end

function [output,loss,model,gradloss] = lossOfModel_adam_rand(model,X,lossfun)
        

        Y = logsig(model.W * X + model.B);

        output = reshape(Y,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;
%         loss = L + sum(model.W.^2,'all')/2;
        
        gradloss.W = dlgradient(loss,model.W);
        gradloss.B = dlgradient(loss,model.B);
        
end

function G = KernelRBF(x1,x2)
% x1,x2  -  n * d Matrix; n - number of samples, d - number of features

G = zeros(size(x1,1),size(x2,1));
for i = 1:size(x1,1)
    G(i,:) = exp(-sum((x1(i,:) - x2).^2,2))';
end

end