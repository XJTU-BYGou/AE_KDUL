classdef ttlLossClassificationLayer_featureInput_randavgIter < nnet.layer.ClassificationLayer
    % Example custom classification layer with custom loss.
    properties
        % (Optional) Layer properties.
        timeMat
        % Layer properties go here.
    end
    
    
    methods
        function layer = ttlLossClassificationLayer_featureInput_randavgIter(name,timeMat)
            % layer = ttlLossClassificationLayer_featureInput_randavgIter(name,timeMat) creates a custom
            % error classification layer and specifies the layer name.
    
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'time transfer loss';
            
            % Set inputs.
            timeMat = reshape(timeMat,1,[]);
            layer.timeMat = timeMat;
    
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the custome loss between
            % the predictions Y and the training targets T.

            intNum = max(poissrnd(12),5);

            % Define the timeTransMat
            timeTransMat_l = getTimeTransMat(layer.timeMat,intNum,[],[]);
            l = getAggregateTrendLoss(timeTransMat_l,Y);  
        end
        
    end
end