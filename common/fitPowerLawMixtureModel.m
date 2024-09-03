function [Mdl] = fitPowerLawMixtureModel(x,k,varargin)
% [Mdl] = fitPowerLawMixtureModel(x,k,varargin)
%
%
%


x = reshape(x,1,[]);
if nargin < 1
    error('Too many or too few input arguments.');
elseif nargin == 1
    k = 2;
end

if length(x) < 2
    error('Too few data.')
end

%% Default Parameters
initMethod = 'randSample';
xmin = min(x);
probTol = 1e-8;
regVal = 0;
MaxIter = 1e2;
TolFun = 1e-6;
optDetail = 'detail';

%% Set Name-Value Parameters
for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'initmethod'
            initMethod = varargin{i*2};
        case 'xmin'
            xmin = varargin{i*2};
        case 'probabilitytolerance'
            probTol = varargin{i*2};
        case 'regularizationvalue'
            regVal = varargin{i*2};
        case 'maxiter'
            MaxIter = varargin{i*2};
        case 'tolfun'
            TolFun = varargin{i*2};
        case 'optdetail'
            optDetail = varargin{i*2};
    end
end

%% Init model
initParam = initPLMMdl(x,k,xmin,initMethod);

omega = initParam.Omega;
alpha = initParam.Alpha;
mask = x >= xmin;
llh = nan(MaxIter,1);
for i = 1:MaxIter
%% E-Step
    lh = omega.*(alpha-1)./xmin.*(x./xmin).^-alpha;
    lhs = sum(lh,1);
    gamma = lh./lhs;
    gamma(gamma<probTol) = 0;
    llh(i) = -sum(log(lhs(:,mask)),2);
    if i > 1
        if llh(i) < TolFun  || llh(i-1) - llh(i) < TolFun
            break;
        end
    end
%% M-Step
    omega = mean(gamma(:,mask),2);
    alpha = 1 + sum(gamma(:,mask),2)./sum(gamma(:,mask).*log(x(:,mask)./xmin),2);
end
%% Construct Model

Mdl.InitSetting = initParam;
Mdl.Omega = omega;
Mdl.Alpha = alpha;
Mdl.Loglikelihood = llh(i);
Mdl.XMin = xmin;
Mdl.ProbabilityTolerance = probTol;
Mdl.RegularizationValue = regVal;
Mdl.MaxItertion = MaxIter;
Mdl.ToleranceValue = TolFun;

if strcmpi(optDetail,'detail')
    Mdl.Gamma = gamma;
    Mdl.LLHChange = llh(1:i);
    Mdl.Likelihood = lh;
    Mdl.Mask = mask;
    [~,predLabel] = max(gamma);
    Mdl.Label = predLabel;
    Mdl.X = x;
end

end


function initParam = initPLMMdl(x,k,xmin,initMethod)
if isvector(initMethod) && numel(initMethod) == numel(x)
    initMethod = reshape(initMethod,[],1);
    label = unique(initMethod);
    onehotlabel = initMethod' == label;
    mask = x >= xmin;
    initParam.Omega = mean(onehotlabel(:,mask),2);
    initParam.Alpha = 1 + sum(onehotlabel(:,mask),2)./sum(onehotlabel(:,mask).*log(x(:,mask)./xmin),2);
    initParam.Method = 'custom';
elseif isstruct(initMethod)
    initParam = initMethod;
    initParam.Method = 'custom';
elseif iscell(initMethod)
    initParam.Omega = initMethod{1};
    initParam.Alpha = initMethod{2};
    initParam.Method = 'custom';
elseif ischar(initMethod)
switch lower(initMethod)
    case 'randsample'
        initParam = initParam_randSample(x,k,xmin);
        initParam.Method = 'randSample';
%     case 'kmeans++'
%         initParam = initParam_plus(x,k,xmin);
%         initParam.Method = 'kmeans++';
%     case 'plus'
%         initParam = initParam_plus(x,k,xmin);
%         initParam.Method = 'kmeans++';
end
else
    error('Initialization failed. Method Error.');
end

initParam.Omega = reshape(initParam.Omega,[],1);
initParam.Alpha = reshape(initParam.Alpha,[],1);
if numel(initParam.Omega) ~= k || numel(initParam.Alpha) ~= k || ...
        ~isvector(initParam.Omega) || ~isvector(initParam.Alpha)
    error('Initialization failed. Data structure error.');
end
if any(initParam.Omega<0) || any(initParam.Omega>1) || ...
       sum(initParam.Omega)~=1 || any(initParam.Alpha<1)
    error('Initialization failed. The initialization value error.');
end

%% Sub Function
    function initParam = initParam_randSample(x,k,xmin)
        idx = randsample(k,numel(x),true);
        onehotlabel = (idx' == [1:k]');
        mask = x >= xmin;
        initParam.Omega = mean(onehotlabel(:,mask),2);
        initParam.Alpha = 1 + sum(onehotlabel(:,mask),2)./sum(onehotlabel(:,mask).*log(x(:,mask)./xmin),2);
    end
    function initParam = initParam_plus(x,k,xmin)
        
    end

end