function [superJerkIDvec_all,expMLEstimator_all,enyExpFlag_all,err_all] = autoSuperjerkEstimtor(res_pri_crack,plotFlag,revFlag)
if nargin == 1
    plotFlag = true;
    revFlag = false;
elseif nargin == 2
    revFlag = false;
end


colorRGB1 = [0,43,128]/255;
colorRGB2 = [255 165 0]/255;
colorRGB2_dark = [255,104,40]/255;
colorRGB3 = [0,87,55]/255;



initID = 1;
superJerkID = find([res_pri_crack(1:initID).Eny] == max([res_pri_crack(1:initID).Eny]),1);
superJerkEny = [res_pri_crack(superJerkID).Eny];


ExMat = [];
epsilonMat = [];
errbarMat = [];
clear enyExpFlag
keysuperJerkID = [];

expMLEstimator = nan;
superJerkIDvec = superJerkID;
enyExpFlag = false;
err = nan;

itr = 1;
while ~isempty(find([res_pri_crack.Eny]>superJerkEny & [1:numel(res_pri_crack)]>superJerkID,1))
    itr = itr + 1;
    superJerkID_new = find([res_pri_crack.Eny]>superJerkEny & [1:numel(res_pri_crack)]>superJerkID,1);
    superJerkEny = [res_pri_crack(superJerkID_new).Eny];
    
    
    superJerkIDvec(itr) = superJerkID_new;

    interEny = [res_pri_crack(superJerkID+1:superJerkID_new).Eny];
    
    MLEny = interEny;    
    if numel(MLEny) < 5 %50 
        if superJerkID_new > 4 %49
            MLEny = [res_pri_crack(superJerkID_new-4:superJerkID_new).Eny];
            enyExpFlag(itr) = true;
        else
            enyExpFlag(itr) = false;
            expMLEstimator(itr) = nan;
            err(itr) = nan;
            superJerkID = superJerkID_new;
        end
    else
        enyExpFlag(itr) = true;
    end
    
    if enyExpFlag(itr)
        [res,record] = powerlawExponentMLEstimator(MLEny);
        expMLEstimator(itr) = res.Exponent;
        err(itr) = res.Err;
    end
    superJerkID = superJerkID_new;
    
    
    if plotFlag
        cumEny = [res_pri_crack(1:superJerkID_new).Eny];
        [inres,inrecord] = powerlawExponentMLEstimator(interEny);
        [cres,crecord] = powerlawExponentMLEstimator(cumEny);
        
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
            
        p1 = errorbar(inrecord.Xmin,inrecord.Alpha,inrecord.Err,'o','LineWidth',1.5);
        p0 = errorbar(record.Xmin,record.Alpha,record.Err,'s','LineWidth',1.5);
        p2 = errorbar(crecord.Xmin,crecord.Alpha,crecord.Err,'d','LineWidth',1.5);
        l1 = plot([inres.Xmin,inrecord.Xmin(end)],inres.Exponent.*[1,1],'Color',p1.Color,'LineWidth',2);
        l0 = plot([res.Xmin,record.Xmin(end)],res.Exponent.*[1,1],'Color',p0.Color,'LineWidth',2);        
        l2 = plot([cres.Xmin,crecord.Xmin(end)],cres.Exponent.*[1,1],'Color',p2.Color,'LineWidth',2);
        hl = legend([p1,p2,p0],{[num2str([res_pri_crack(superJerkID).Time]),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s'],...
            [num2str(stTime),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s'],...
            ['MLE']},...
            'FontName','Arial','FontSize',12,'FontWeight','bold');
        legend('boxoff');

    end
   
    
end

if revFlag
%% 反向计算
maxSuperJerkID = superJerkIDvec(end);
superJerkEny = res_pri_crack(end).Eny;
superJerkID = numel(res_pri_crack);

clear enyExpFlag_rev

expMLEstimator_rev = nan;
superJerkIDvec_rev = superJerkID;
enyExpFlag_rev = false;
err_rev = nan;

itr = 1;
while ~isempty(find([res_pri_crack.Eny]>superJerkEny & [1:numel(res_pri_crack)]<superJerkID,1,'last'))
    itr = itr + 1;
    superJerkID_new = find([res_pri_crack.Eny]>superJerkEny & [1:numel(res_pri_crack)]<superJerkID,1,'last');
    superJerkEny = [res_pri_crack(superJerkID_new).Eny];
    
    superJerkIDvec_rev(itr) = superJerkID_new;
    interEny = [res_pri_crack(superJerkID_new+1:superJerkID).Eny];
    
    
    MLEny = interEny;    
    if numel(MLEny) < 5 % 50
        if numel(res_pri_crack) - superJerkID_new > 5 % 50
            MLEny = [res_pri_crack(superJerkID_new+1:superJerkID_new+5).Eny];
            enyExpFlag_rev(itr) = true;
        else
            enyExpFlag_rev(itr) = false;
            expMLEstimator_rev(itr) = nan;
            err_rev(itr) = nan;
            superJerkID = superJerkID_new;
        end
    else
        enyExpFlag_rev(itr) = true;
    end
    
    if enyExpFlag_rev(itr)
        [res,record] = powerlawExponentMLEstimator(MLEny);
        expMLEstimator_rev(itr) = res.Exponent;
        err_rev(itr) = res.Err;
    end
    superJerkID = superJerkID_new;

    
    if plotFlag
        cumEny = [res_pri_crack(superJerkID_new+1:end).Eny];
        [inres,inrecord] = powerlawExponentMLEstimator(interEny);
        [cres,crecord] = powerlawExponentMLEstimator(cumEny);
        
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

        p1 = errorbar(inrecord.Xmin,inrecord.Alpha,inrecord.Err,'o','LineWidth',1.5);
        p0 = errorbar(record.Xmin,record.Alpha,record.Err,'s','LineWidth',1.5);
        p2 = errorbar(crecord.Xmin,crecord.Alpha,crecord.Err,'d','LineWidth',1.5);
        l1 = plot([inres.Xmin,inrecord.Xmin(end)],inres.Exponent.*[1,1],'Color',p1.Color,'LineWidth',2);
        l0 = plot([res.Xmin,record.Xmin(end)],res.Exponent.*[1,1],'Color',p0.Color,'LineWidth',2);        
        l2 = plot([cres.Xmin,crecord.Xmin(end)],cres.Exponent.*[1,1],'Color',p2.Color,'LineWidth',2);
        hl = legend([p1,p2,p0],{[num2str([res_pri_crack(superJerkID).Time]),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s'],...
            [num2str(stTime),'s -> ',num2str([res_pri_crack(superJerkID_new).Time]),'s'],...
            ['MLE']},...
            'FontName','Arial','FontSize',12,'FontWeight','bold');
        legend('boxoff');

    end

end


superJerkIDvec_all = [superJerkIDvec,fliplr(superJerkIDvec_rev(1:end-1)),numel(res_pri_crack)];
enyExpFlag_all = [enyExpFlag,fliplr(enyExpFlag_rev)];
expMLEstimator_all = [expMLEstimator,fliplr(expMLEstimator_rev)];
err_all = [err,fliplr(err_rev)];

else

superJerkIDvec_all = [superJerkIDvec];
enyExpFlag_all = [enyExpFlag];
expMLEstimator_all = [expMLEstimator];
err_all = [err];
end

end

%%
function [exp] = estimateExponentFromMLData(Ex,epsilon)
% get the estimated exponent
%% Median estimate
[~,index] = min(abs(Ex-median(Ex)));
exp = epsilon(index(1));
%% raw data
% x = log10(Ex);
% k = abs(diff(epsilon)./diff(x));
% exp = sum(epsilon(1:end-1).*10.^(-k))./sum(10.^(-k));
%% interp1
% [vx,vy] = interp1forMLData(Ex,epsilon);
% x = log10(vx);
% k = abs(diff(vy)./diff(x));
% exp = sum(vy(1:end-1).*10.^(-k))./sum(10.^(-k));
%% dis sqrt
% exp = 1;
% lr = 1e-2;
% expGredient = 0;
% while expGredient < 1e6
%     lr = 1e-3;
%     exp0 = exp;
%     expGredient = mean(sign(exp-epsilon).*1./sqrt(abs(epsilon - exp)))./2;
%     exp = exp - expGredient .* lr;
% end

end
function [vx,vy] = interp1forMLData(Ex,epsilon)
x = log10(Ex);
n = 200;
vx = min(x):(max(x)-min(x))/200:max(x);
vy = interp1(x,epsilon,vx);
vx = 10.^vx;
end