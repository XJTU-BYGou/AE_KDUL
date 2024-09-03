function [superJerkIDvec_all,expMLEstimator_all,enyExpFlag_all,err_all] = superjerkEstimtor_insuit(res_pri_crack,plotFlag)
if nargin < 2
    plotFlag = true;
end

colorRGB1 = [0,43,128]/255;
colorRGB2 = [255 165 0]/255;
colorRGB2_dark = [255,104,40]/255;
colorRGB3 = [0,87,55]/255;



initID = 1;
superJerkID = find([res_pri_crack(1:initID).Eny] == max([res_pri_crack(1:initID).Eny]),1);
superJerkEny = [res_pri_crack(superJerkID).Eny];
superJerkIDvec = [superJerkID];

ExMat = [];
epsilonMat = [];
errbarMat = [];
clear enyExpFlag
keysuperJerkID = [];

itr = 0;
while ~isempty(find([res_pri_crack.Eny]>superJerkEny & [1:numel(res_pri_crack)]>superJerkID,1)) && itr < 20
    itr = itr + 1;
    superJerkID_new = find([res_pri_crack.Eny]>superJerkEny & [1:numel(res_pri_crack)]>superJerkID,1);
    superJerkEny = [res_pri_crack(superJerkID_new).Eny];
    
    interEny = [res_pri_crack(superJerkID+1:superJerkID_new).Eny];
    
    if isempty(keysuperJerkID)
        cumEny = [res_pri_crack(1:superJerkID_new).Eny];
        stTime = 0;
    else
        cum0Eny = [res_pri_crack(1:superJerkID_new).Eny];
        cumEny = [res_pri_crack(keysuperJerkID+1:superJerkID_new).Eny];
        stTime = res_pri_crack(keysuperJerkID).Time;
    end
    
    if numel(interEny) < 5
        superJerkIDvec(itr) = superJerkID_new;
        superJerkID = superJerkID_new;
        continue
    end
    [epsilon,errbar,Ex] = aeMLEstimator(interEny);
    [cepsilon,cerrbar,cEx] = aeMLEstimator(cumEny);
    [res] = powerlawExponentMLEstimator(interEny);
    err(itr) = res.Err;
    expMLEstimator(itr) = estimateExponentFromMLData(Ex,epsilon);
    
    if plotFlag
    fig = figure;
    fig.Position = [60,270,700,420];
    ax = axes('Units','pixels','Position',[120 80 500 300]);
    ax.XScale = 'log';
    ax.Box = 'on';
    hold on;
    ylim([1,4]);
    set(gca,'FontName','Arial','FontSize',20,'FontWeight','bold');
    ylabel([char(949)],'FontName','Arial','FontSize',22,'FontWeight','bold');
    xlabel(['Energy (aJ)'],'FontName','Arial','FontSize',22,'FontWeight','bold');
    set(gca,'color','none');
    ax.LineWidth = 2;
    ax.TickLength = [0.02,0.05];
    ax.XTick = 10.^[-2:6];
    ax.YTick = [1:0.5:6];
    ax.YMinorTick = 'on';
    ax.YAxis.MinorTickValues = 1:0.1:6;
    if isempty(keysuperJerkID)
        p1 = errorbar(Ex,epsilon,errbar,'o','LineWidth',1.5);
        p2 = errorbar(cEx,cepsilon,cerrbar,'d','LineWidth',1.5);
        hl = legend([p1,p2],{[num2str([res_pri_crack(superJerkID).Time]),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s'],...
            [num2str(stTime),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s']},...
            'FontName','Arial','FontSize',12,'FontWeight','bold');
        legend('boxoff');
    else
        [c0epsilon,c0errbar,c0Ex] = aeMLEstimator(cum0Eny);
        p1 = errorbar(Ex,epsilon,errbar,'o','LineWidth',1.5);
        p0 = errorbar(c0Ex,c0epsilon,c0errbar,'s','LineWidth',1.5);
        p2 = errorbar(cEx,cepsilon,cerrbar,'d','LineWidth',1.5);
        hl = legend([p1,p2,p0],{[num2str([res_pri_crack(superJerkID).Time]),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s'],...
            [num2str(stTime),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s'],...
            ['0s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s']},...
            'FontName','Arial','FontSize',12,'FontWeight','bold');
        legend('boxoff');
    end
    end
    if expMLEstimator(itr) < 1.6 && isempty(keysuperJerkID)
        keysuperJerkID = superJerkID_new;
        keyTimePoint = res_pri_crack(superJerkID_new).Time;
    end
    
    enyExpFlag(itr) = true;
    superJerkIDvec(itr) = superJerkID_new;
    superJerkID = superJerkID_new;
    
    ExMat = [ExMat,{Ex}];
    epsilonMat = [epsilonMat,{epsilon}];
    errbarMat = [errbarMat,{errbar}];
end

superJerkIDvec_all = [superJerkIDvec];
enyExpFlag_all = [enyExpFlag];
expMLEstimator_all = [expMLEstimator];
err_all = [err];
end

%%
function [exp] = estimateExponentFromMLData(Ex,epsilon)
[~,index] = min(abs(Ex-median(Ex)));
exp = epsilon(index(1));
end
function [vx,vy] = interp1forMLData(Ex,epsilon)
x = log10(Ex);
n = 200;
vx = min(x):(max(x)-min(x))/200:max(x);
vy = interp1(x,epsilon,vx);
vx = 10.^vx;
end

function [epsilon,errbar,Ex,id] = aeMLEstimator(Eny,E0)
% Get the epsilon with the Maximum Likelihood ep.
%   By Gby 2019-03-27
%   Power by MATLAB 2020a
%   Last change on 2020-11-27
if nargin < 1 || nargin > 2
    error('Too many or too few input arguments.');
elseif nargin == 1
    E0 = [];
end
    
if length(Eny) < 2
    %error('Too few Eny data.')
    epsilon = [];
    errbar = [];
    Ex = [];
    id = [];
else


if isempty(E0) 
    [Eny,id] = sort(Eny);
for i = length(Eny)-1:-1:1
    tmp = log(Eny(i+1:end)/Eny(i));
    epsilon(i) = 1 + 1/mean(tmp);
    errbar(i) = sqrt(var(tmp)./length(tmp))./mean(tmp).^2;
end
id = id(1:end-1);
Ex = Eny(1:end-1);
else
%% Special E0
id = [];
epsilon = [];
errbar = [];
for i = length(E0):-1:1
    if ~isempty(Eny(Eny>E0(i)))
    tmp = log(Eny(Eny>E0(i))/E0(i));
    if ~isempty(tmp)
        epsilon = [1 + 1/mean(tmp),epsilon];
        errbar = [sqrt(var(tmp)./length(tmp))./mean(tmp).^2,errbar];
        id = [i,id];
    end
    end
end
Ex = E0(id);
end
end

Ex = reshape(Ex,1,[]);
epsilon = reshape(epsilon,1,[]);
errbar = reshape(errbar,1,[]);

end