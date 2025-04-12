function [model,output,history,options] = trainGaussianDiscriminative_nongrad(model,...
    XTrain,YTrain,lossfun,varargin)
rng(0);
% Check the input arguments
if nargin < 3
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end
delta = 1e-10;

% Get Properties
% MaxEpoch = 2e3;
MaxEpoch = 10e3;
[numSample,numFeature] = size(XTrain);
numClasses = 2;
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
model.GDA = [];
model.GDA.Model = [];
model.GDA.BestModel.Loss = inf;
model.GDA.BestModel.Model = [];

Model.Alpha = ones(numClasses,1)/numClasses;
Model.Mu = zeros(numClasses,numFeature);
Model.Sigma = zeros(numFeature,numFeature,numClasses);
else
Model = model.GDA.Model;
end

tic;
numParam = numel(Model.Alpha) + numel(Model.Mu) + (size(Model.Sigma,1)+1)*size(Model.Sigma,1)/2*size(Model.Sigma,3);
[estParam,exitflag,optProcess] = ga(@(x)lossOfModel_nongrad(x,Model,XTrain,lossfun),numParam,[],[],[],[],[],[],[],options);
Model = vec2mdl(estParam,Model);


model.GDA.Model = Model;
[loss,output] = lossOfModel(Model,XTrain,lossfun);
model.GDA.Loss = loss;
model.GDA.BestModel.Loss = loss;
model.GDA.BestModel.Model = Model;

[~,YL] = max(output);
curLoop.Acc = mean(double(YTrain') == YL);
curLoop.Loss = double(loss);
curLoop.ElapsedTime = toc;
curLoop.OptProcess = optProcess;
curLoop.Finish = exitflag;
history = curLoop;

% for i = 1:options.MaxEpoch
% %     Py = lh./sum(lh);
% 
%     % Calculate custom Loss and gradient
%     [output,loss,gradloss] = dlfeval(@lossOfModel_rand,model,(Py),lossfun);
%     output = extractdata(output);
%     loss = double(extractdata(loss));
% %     Y = extractdata(Y);
% 
%     % Update parameters
%     lr = options.InitialLearnRate;
% 
%     Py = Py - lr .* extractdata(gradloss.X);
%     Py = min(max(Py,0),1);
%     Py = Py./sum(Py);
% 
%     % EM Method
%     Model.Alpha = mean(Py,2);
%     Model.Mu = Py * XTrain ./sum(Py,2);
%     invtau = zeros(numFeature,numFeature,numClasses);
%     for j = 1:numClasses
%         Model.Sigma(:,:,j) = Py(j,:).*(XTrain - Model.Mu(j,:))' ...
%             *(XTrain - Model.Mu(j,:)) ./sum(Py(j,:),2) ;
%         
%         invtau(:,:,j) = inv(Model.Sigma(:,:,j));
%         dettau(j,1) = det(Model.Sigma(:,:,j)) + 0;
%     end
%     
%     dim = numFeature;
%     lhe = gpuArray(zeros(size(Py)));
%     lhe = [diag(Model.Alpha(1).*exp(-0.5.*...
%             (XTrain-Model.Mu(1,:))*invtau(:,:,1)*(XTrain- Model.Mu(1,:))')...
%             ./sqrt((2*pi)^dim*dettau(1)))';
%             diag(Model.Alpha(2).*exp(-0.5.*...
%             (XTrain-Model.Mu(2,:))*invtau(:,:,2)*(XTrain- Model.Mu(2,:))')...
%             ./sqrt((2*pi)^dim*dettau(2)))'];
%     
%     lhe = lhe + delta;
%     output = lhe./sum(lhe);
%     if mod(i,1) == 0
%         lh = lhe;
%         Py = lh./sum(lh);
%     end
%     
%     model.GDA.Model = Model;
%     model.GDA.Loss = loss;
%     if loss < model.GDA.BestModel.Loss
%         model.GDA.BestModel.Loss = loss;
%         model.GDA.BestModel.Model = Model;
%     end
%     
%     %% Record the processing
%     [~,YL] = max(output);
%     curLoop.Acc = mean(double(YTrain') == YL);
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

function [output,loss,gradloss] = lossOfModel_rand(model,X,lossfun)
        
        output = X;
        L = lossfun(output);
        
        loss =  L;
        gradloss.X = dlgradient(loss,X);

end

function [loss] = lossOfModel_nongrad(param,Model,XTrain,lossfun)
        Model = vec2mdl(param,Model);
        Y = gmmCalcu(Model,XTrain);
        
        output = Y;
        L = lossfun(output);
        
        loss =  L;
end
function [loss,output] = lossOfModel(Model,XTrain,lossfun)
        
        Y = gmmCalcu(Model,XTrain);
        
        output = Y;
        L = lossfun(output);
        
        loss =  L;
end

function param = mdl2vec(Model)
param = [Model.Alpha(:);Model.Mu(:);Model.Sigma(:)];
end

function Model = vec2mdl(param,Model)
[numFeature] = size(Model.Sigma,1);
numClasses = size(Model.Mu,1);

Model.Alpha = reshape(param(1:numClasses),numClasses,1);
Model.Mu = reshape(param(numClasses+1:numClasses+numFeature*numClasses),size(Model.Mu));

sigmaInd = [1:numFeature]'+[0:numFeature-1].*numFeature - [0:numFeature-1].*[1:numFeature]/2;
sigmaInd = tril(sigmaInd) + tril(sigmaInd,-1)';

SigmaParam = reshape(param(numClasses+numFeature*numClasses+1:end),[],size(Model.Sigma,3));
for i = 1:numClasses
    tmp = SigmaParam(:,i);
    Model.Sigma(:,:,i) = tmp(sigmaInd);
end
end


function Py = gmmCalcu(Model,XTrain)
[numSample,numFeature] = size(XTrain);
numClasses = size(Model.Mu,1);
invtau = zeros(numFeature,numFeature,numClasses);
for j = 1:numClasses
    invtau(:,:,j) = inv(Model.Sigma(:,:,j));
    dettau(j,1) = det(Model.Sigma(:,:,j)) + 0;
end

dim = numFeature;
lh = [diag(Model.Alpha(1).*exp(-0.5.*...
        (XTrain-Model.Mu(1,:))*invtau(:,:,1)*(XTrain- Model.Mu(1,:))')...
        ./sqrt((2*pi)^dim*dettau(1)))';
        diag(Model.Alpha(2).*exp(-0.5.*...
        (XTrain-Model.Mu(2,:))*invtau(:,:,2)*(XTrain- Model.Mu(2,:))')...
        ./sqrt((2*pi)^dim*dettau(2)))'];
lh = lh + 1e-8;
Py = lh./sum(lh);
end