classdef lossOnlyTrend < nnet.layer.ClassificationLayer
    properties
        timeMat_p1
        timeMat_p2
        preIndex
        timeMat
        
        alpha = 0.5;
        gamma = 0;
        w = 0.5;
    end
    
    
    methods
        function layer = lossOnlyTrend(name,timeMat1,timeMat2,preIndex,timeMat,gamma,w)
    
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'time transfer loss';
            
            % Set inputs.
            timeMat1 = reshape(timeMat1,1,[]);
            layer.timeMat_p1 = timeMat1;
            timeMat2 = reshape(timeMat2,1,[]);
            layer.timeMat_p2 = timeMat2;     
            layer.preIndex = preIndex;
            layer.timeMat = timeMat;
            
            layer.gamma = gamma;
            layer.w = w;
        end
        
        function loss = forwardLoss(layer, Y, T)

            intNum = max(poissrnd(12),5);

            % Define the timeTransMat
            timeTransMat_l = getTimeTransMat(layer.timeMat,intNum,[],layer.timeInt);
            l = getAggregateTrendLoss(timeTransMat_l,Y,'w',layer.w);  
        end
           
    end
end