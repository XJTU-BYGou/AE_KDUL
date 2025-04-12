function [model,output,history,options] = trainGaussianDiscriminative_rand(model,...
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
initialLearnRate = 1e1;
gradientDecayFactor = 0.9;
learnRateDropPeriod = 1e4;
learnRateDropFactor = 0.8;
UpdatePeriod = 100;
Verbose = true;
VerboseFrequency = 50;

options.InitialLearnRate = initialLearnRate;
options.GradientDecayFactor = gradientDecayFactor;
options.LearnRateDropPeriod = learnRateDropPeriod;
options.LearnRateDropFactor = learnRateDropFactor;
options.UpdatePeriod = UpdatePeriod;
options.Verbose = Verbose;
options.VerboseFrequency = VerboseFrequency;
options.MaxEpoch = MaxEpoch;

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'options'
        options = varargin{i*2};
    end
end

if ~isfield(options,'UpdatePeriod')
    options.UpdatePeriod = UpdatePeriod;
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

[idx] = kmeans(XTrain,numClasses,'distance','cosine');
idx = randsample(numClasses,numSample,true);
Py = double(idx' == [1:numClasses]');

% EM Method
Model.Alpha = mean(Py,2);
Model.Mu = Py * XTrain ./sum(Py,2);
invtau = zeros(numFeature,numFeature,numClasses);
for j = 1:numClasses
    Model.Sigma(:,:,j) = Py(j,:).*(XTrain - Model.Mu(j,:))' ...
        *(XTrain - Model.Mu(j,:)) ./sum(Py(j,:),2) ;

    invtau(:,:,j) = inv(Model.Sigma(:,:,j));
    dettau(j,1) = det(Model.Sigma(:,:,j)) + 0;
end

dim = numFeature;
lh = gpuArray(zeros(size(Py)));
lh = [diag(Model.Alpha(1).*exp(-0.5.*...
        (XTrain-Model.Mu(1,:))*invtau(:,:,1)*(XTrain- Model.Mu(1,:))')...
        ./sqrt((2*pi)^dim*dettau(1)))';
        diag(Model.Alpha(2).*exp(-0.5.*...
        (XTrain-Model.Mu(2,:))*invtau(:,:,2)*(XTrain- Model.Mu(2,:))')...
        ./sqrt((2*pi)^dim*dettau(2)))'];
% for j = 1:numSample
%     lh(:,j) = [Model.Alpha(1).*exp(-0.5.*...
%         (XTrain(j,:)-Model.Mu(1,:))*invtau(:,:,1)*(XTrain(j,:) - Model.Mu(1,:))')...
%         ./sqrt((2*pi)^dim*dettau(1));...
%                Model.Alpha(2).*exp(-0.5.*...
%         (XTrain(j,:)-Model.Mu(2,:))*invtau(:,:,2)*(XTrain(j,:) - Model.Mu(2,:))')...
%         ./sqrt((2*pi)^dim*dettau(2))];
% end
lh = lh + delta;
Py = lh./sum(lh);

for i = 1:options.MaxEpoch
%     Py = lh./sum(lh);

    % Calculate custom Loss and gradient
    [output,loss,gradloss] = dlfeval(@lossOfModel_rand,model,dlarray(Py),lossfun);
    output = extractdata(output);
    loss = double(extractdata(loss));
%     Y = extractdata(Y);

    model.GDA.Model = Model;
    model.GDA.Loss = loss;

    % Update parameters
    lr = options.InitialLearnRate;
    
%     lh = lh - lr .* extractdata(gradloss.X);
%     lh = max(lh,0);
%     Py = lh./sum(lh);
%     Py = exp(lh)./sum(exp(lh));

    Py = Py - lr .* extractdata(gradloss.X);
    Py = min(max(Py,0),1);
    Py = Py./sum(Py);
    

    
if mod(i,options.UpdatePeriod) == 0 || i == options.MaxEpoch
    PPy = Py;
%     [~,Y] = max(Py);
%     PPy = Y == [1:numClasses]';
    
    % EM Method
    Model.Alpha = mean(PPy,2);
    Model.Mu = PPy * XTrain ./sum(PPy,2);
    invtau = zeros(numFeature,numFeature,numClasses);
    for j = 1:numClasses
        Model.Sigma(:,:,j) = PPy(j,:).*(XTrain - Model.Mu(j,:))' ...
            *(XTrain - Model.Mu(j,:)) ./sum(PPy(j,:),2) ;
        
        invtau(:,:,j) = inv(Model.Sigma(:,:,j));
        dettau(j,1) = det(Model.Sigma(:,:,j)) + 0;
    end
    
%     sigma = zeros(size(Model.Sigma(:,:,1)));
%     for j = 1:numClasses
%         sigma = sigma + Model.Sigma(:,:,j) .* sum(PPy(j,:),2);
%     end
%     signa = sigma./size(PPy,2);
%     for j = 1:numClasses
%         Model.Sigma(:,:,j) = signa;
%     end
    
    dim = numFeature;
    lhe = gpuArray(zeros(size(PPy)));
    lhe = [diag(Model.Alpha(1).*exp(-0.5.*...
            (XTrain-Model.Mu(1,:))*invtau(:,:,1)*(XTrain- Model.Mu(1,:))')...
            ./sqrt((2*pi)^dim*dettau(1)))';
            diag(Model.Alpha(2).*exp(-0.5.*...
            (XTrain-Model.Mu(2,:))*invtau(:,:,2)*(XTrain- Model.Mu(2,:))')...
            ./sqrt((2*pi)^dim*dettau(2)))'];
%     for j = 1:numSample
%         lhe(:,j) = [Model.Alpha(1).*exp(-0.5.*...
%             (XTrain(j,:)-Model.Mu(1,:))*invtau(:,:,1)*(XTrain(j,:) - Model.Mu(1,:))')...
%             ./sqrt((2*pi)^dim*dettau(1));...
%                     Model.Alpha(2).*exp(-0.5.*...
%             (XTrain(j,:)-Model.Mu(2,:))*invtau(:,:,2)*(XTrain(j,:) - Model.Mu(2,:))')...
%             ./sqrt((2*pi)^dim*dettau(2))];
%     end
    
    lhe = lhe + delta;
    output = lhe./sum(lhe);
    
        lh = lhe;
        Py = lh./sum(lh);
        
    [output,loss,~] = dlfeval(@lossOfModel_rand,model,dlarray(Py),lossfun);
    output = extractdata(output);
    loss = double(extractdata(loss));
    
    model.GDA.Model = Model;
    model.GDA.Loss = loss;
    if loss < model.GDA.BestModel.Loss
        model.GDA.BestModel.Loss = loss;
        model.GDA.BestModel.Model = Model;
    end
end
    
    %% Record the processing
    [~,YL] = max(output);
    curLoop.Acc = mean(double(YTrain') == YL);
    curLoop.Loss = loss;
    curLoop.ElapsedTime = toc;
    history(i,1) = curLoop;
    if options.Verbose
        if mod(i,options.VerboseFrequency) == 0 || i == 1
            fprintf('Epoch: %i , Training Time: %f , Loss: %f , Accuracy: %f \n',...
                i,curLoop.ElapsedTime,curLoop.Loss,curLoop.Acc);
        end
        if i == 1
            fig = figure('Position',[200,150,680,800]);
            axbg = axes(fig,'Units','pixels','Position',[100 460 500 300],...
            'Color', 'none','Box','off',...
            'XAxisLocation','top','YAxisLocation','right',...
            'LineWidth',2,'TickLength', [0.02,0.05],...
            'XTick',[],'YTick',[]);
            ax1 = axes(fig,'Units','pixels','Position',axbg.Position,...
            'Color', 'none','Box','off',...
            'LineWidth',2,'TickLength', [0.02,0.05],...
            'FontName','Arial','FontSize',16,'FontWeight','bold');    
            xlabel('Iteration');
            ylabel('Pseudo Accuracy (%)');
            ax1.YLim = [0,100];
            hold on;
            anAcc = animatedline(ax1,i,curLoop.Acc.*100,'Color','b','LineWidth',2);
        
            axbg = axes(fig,'Units','pixels','Position',[100 80 500 300],...
            'Color', 'none','Box','off',...
            'XAxisLocation','top','YAxisLocation','right',...
            'LineWidth',2,'TickLength', [0.02,0.05],...
            'XTick',[],'YTick',[]);
            ax2 = axes(fig,'Units','pixels','Position',axbg.Position,...
            'Color', 'none','Box','off',...
            'LineWidth',2,'TickLength', [0.02,0.05],...
            'FontName','Arial','FontSize',16,'FontWeight','bold');    
            hold on;
            xlabel('Iteration');
            ylabel('Loss');
            anLoss = animatedline(ax2,i,curLoop.Loss,'Color','r','LineWidth',2);
            drawnow();
        else
            addpoints(anAcc,i,curLoop.Acc.*100);
            addpoints(anLoss,i,curLoop.Loss);
            drawnow();
        end
    end
end
end

function [output,loss,gradloss] = lossOfModel_rand(model,X,lossfun)
        
        
%         output = exp(X)./sum(exp(X));
%         output = X./sum(X);
        output = X;
        L = lossfun(output);
        
        loss =  L;
%         loss = L + sum(model.W.^2,'all')/2;
        

        gradloss.X = dlgradient(loss,X);


end

