function [model,output,history,options] = trainKernelSupportVectorMachine_rand_v2(model,...
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
model.KSVM = [];
model.KSVM.Model = [];
model.KSVM.BestModel.Loss = inf;
model.KSVM.BestModel.Model = [];
    
W = dlarray(zeros(numClasses,numSample));
B = dlarray(zeros(numClasses,1));
Alpha = dlarray(zeros(numClasses,numSample));
A = dlarray([randn(numClasses,1);0]);

Model.weights.B = B;
Model.weights.Alpha = Alpha;
Model.weights.A = A;

Model.Y = zeros(size(Model.weights.Alpha));
else
Model = model.KSVM.Model;

end
averageGrad = [];
averageSqGrad = [];


tic;

for i = 1:options.MaxEpoch
   
    Y = Model.weights.Alpha .* Model.Y * G + Model.weights.B;
    
    YL = logsig(Y.*Model.weights.A(1,1)+Model.weights.A(2,1));
    % Calculate custom Loss and gradient
    [output,loss,gradloss] = dlfeval(@lossOfModel_rand,Model,YL,lossfun);
    output = extractdata(output);
    loss = extractdata(loss);
    Y = extractdata(Y);

    % Update parameters
    lr = initialLearnRate;
    
    YL_pre = extractdata(sign(-gradloss.X));
    

    sv = (YL_pre .* (Y.*Model.weights.A(1,1)+Model.weights.A(2,1))) <= 0;
    batchFlag = rand(size(Model.weights.Alpha)) < 1;
    grad.weights.Alpha = 0.*Model.weights.Alpha - batchFlag.* (sv);
    grad.weights.B = - (batchFlag.* sv) * YL_pre'./(sum(sv)+1e-6);
    
    grad.weights.A(1,1) = sum(batchFlag.* gradloss.X.*YL.*(1-YL).*Y);
    grad.weights.A(2,1) = sum(batchFlag.* gradloss.X.*YL.*(1-YL));
    
    [Model.weights,averageGrad,averageSqGrad] = adamupdate(Model.weights,grad.weights,averageGrad,averageSqGrad,i,lr);
    
    Y = Model.weights.Alpha .* YL_pre * G + Model.weights.B;
    YL = logsig(Y.*Model.weights.A(1,1)+Model.weights.A(2,1));
    Model.Y = sign(YL-0.5);
    
    
    
    model.KSVM.Model = Model;
    model.KSVM.Loss = loss;
    if loss < model.KSVM.BestModel.Loss
        model.KSVM.BestModel.Loss = loss;
        model.KSVM.BestModel.Model = Model;
    end

    %% Record the processing
    
    curLoop.Acc = mean(double(YTrain')-1 == (output(end,:)>0.5));
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
        
        output = reshape(X,1,[]);
        output = [1-output;output];
        L = lossfun(output);
        
        loss =  L;

        gradloss.X = dlgradient(loss,X);

end

