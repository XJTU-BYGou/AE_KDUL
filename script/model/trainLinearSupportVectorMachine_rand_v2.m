function [model,output,history,options] = trainLinearSupportVectorMachine_rand_v2(model,...
    XTrain,YTrain,lossfun,varargin)
rng(0);
% Check the input arguments
if nargin < 3
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end


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
model.SVM = [];
model.SVM.Model = [];
model.SVM.BestModel.Loss = inf;
model.SVM.BestModel.Model = [];
    
W = dlarray(zeros(numClasses,numFeature));
B = dlarray(zeros(numClasses,1));
A = dlarray([randn(numClasses,1);0]);

Model.weights.W = W;
Model.weights.B = B;
Model.weights.A = A;
else
Model = model.SVM.Model;

end
vel_W = [];
vel_B =[];
averageGrad = [];
averageSqGrad = [];
averageGrad_B = [];
averageSqGrad_B = [];


tic;

for i = 1:options.MaxEpoch
    
    Y = Model.weights.W * XTrain' + Model.weights.B;
    
    YL = logsig(Y.*Model.weights.A(1,1)+Model.weights.A(2,1));
    % Calculate custom Loss and gradient
    [output,loss,gradloss] = dlfeval(@lossOfModel_rand,Model,YL,lossfun);
    output = extractdata(output);
    loss = double(extractdata(loss));

    % Update parameters
    lr = initialLearnRate ;
    
    YL_pre = extractdata(sign(-gradloss.X));
    
    sv = (YL_pre .* (Y.*Model.weights.A(1,1)+Model.weights.A(2,1))) <= 0;
    grad.weights.W = 1e-3.*Model.weights.W - (sv .* YL_pre * XTrain./(sum(sv)+1e-6));
    grad.weights.B = - sv * YL_pre'./(sum(sv)+1e-6);
    
    grad.weights.A(1,1) = sum(gradloss.X.*YL.*(1-YL).*Y);
    grad.weights.A(2,1) = sum(gradloss.X.*YL.*(1-YL));

    [Model.weights,averageGrad,averageSqGrad] = adamupdate(Model.weights,grad.weights,averageGrad,averageSqGrad,i,lr);

    model.SVM.Model = Model;
    model.SVM.Loss = loss;
    if loss < model.SVM.BestModel.Loss
        model.SVM.BestModel.Loss = loss;
        model.SVM.BestModel.Model = Model;
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

