classdef aggregatelosslayer_variant1 < nnet.layer.ClassificationLayer
    properties
        timeMat
        timeInt = [];
        
        w = 0.5;
    end
    
    
    methods
        function layer = aggregatelosslayer_variant1(name,timeMat,varargin)
    
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'loss';
            
            % Set inputs.
            layer.timeMat = timeMat;
            
            for i = 1:length(varargin)/2
                switch lower(varargin{i*2-1})
                    case 'w'
                    layer.w = varargin{i*2};
                    case 'timeint'
                    layer.timeInt = varargin{i*2};
                end
            end
        end
        
        function loss = forwardLoss(layer, Y, T)

%             intNum = max(poissrnd(12),5);
            timeTransMat_l = @(x)getTimeTransMat(layer.timeMat,x,[],layer.timeInt);
            l = getAggregateTrendLoss_variant1(timeTransMat_l,Y,'w',layer.w);    
                        
            loss = l;
        end
        
        
    end
end