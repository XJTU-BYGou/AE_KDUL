function [res,record] = powerlawExponentMLEstimator(X,x0)
% [res,record] = powerlawExponentMLEstimator(X,x0)
% Get the exponent (alpha) with the Maximum Likelihood ep.
%   By Gby 2021-08-05
%   Power by MATLAB 2021a
%   

X = reshape(X,1,[]);
if nargin < 1 || nargin > 2
    error('Too many or too few input arguments.');
elseif nargin == 1 || isempty(x0)
    x0 = unique(X,'sorted');
end


    
if length(X) < 2
    %error('Too few Eny data.')
    res.Xmin  = nan;
    res.Exponent = nan;
    res.Err = nan;
    res.Num_tail = nan;
    res.D = nan;
    
    record.Xmin = nan;
    record.Alpha = nan;
    record.Err = nan;
    record.Dis = nan;
    record.N_tail = nan;
    
else
    
[X,id] = sort(X);
alpha = nan(size(x0));
errbar = nan(size(x0));
Dis = nan(size(x0));
N_tail = nan(size(x0));
    
for i = 1:length(x0)-1
    tmpX = X(X>=x0(i));
    if ~isempty(tmpX)
        tmp = log(tmpX/x0(i));
        n = length(tmp);
        a = 1/mean(tmp);  % alpha = a + 1  <===> a = alpha - 1
        alpha(i) = 1 + a;
        % Err base on the true power law
        errbar(i) = (alpha(i) - 1)/sqrt(length(tmp));
        % Err base on the dataset
%         errbar(i) = sqrt(var(tmp)./length(tmp))./mean(tmp).^2;
        
        % K-S Statistic
        cx   = (0:n-1)./n;
        cf   = 1-(x0(i)./tmpX).^a;
%         Dis(i) = max( abs(cf-cx) );
        Dis(i) = max( abs(cf-cx) ./sqrt(cf.*(1-cf)));
        N_tail(i) = n;
    end
end

% lnX = log(X);
% cumlgX = cumsum(lnX, 'reverse');
% N = cumsum(ones(size(X)), 'reverse');
% A = N./(cumlgX-N.*lnX);
% Alpha = 1 + A;
% Errbar = (Alpha - 1)./sqrt(N);


x0 = x0(~isnan(alpha));
errbar = errbar(~isnan(alpha));
Dis = Dis(~isnan(alpha));
N_tail = N_tail(~isnan(alpha));
alpha = alpha(~isnan(alpha));

if ~isempty(x0)
D = min(Dis);
Id = find(Dis<=D,1,'first');

res.Xmin  = x0(Id);
res.Exponent = alpha(Id);
res.Err = errbar(Id);
res.Num_tail = N_tail(Id);
res.D = D;

record.Xmin = reshape(x0,1,[]);
record.Alpha = reshape(alpha,1,[]);
record.Err = reshape(errbar,1,[]);
record.Dis = reshape(Dis,1,[]);
record.N_tail = reshape(N_tail,1,[]);
else
    res.Xmin  = nan;
    res.Exponent = nan;
    res.Err = nan;
    res.Num_tail = nan;
    res.D = nan;
    
    record.Xmin = nan;
    record.Alpha = nan;
    record.Err = nan;
    record.Dis = nan;
    record.N_tail = nan;
end

end


end