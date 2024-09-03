function [processedData,PCAProcessor] = getPCADataProcess(Data,Thr)

[coeff,score,latent,~,~,mu] = pca(Data);
selectScore = cumsum(latent)/sum(latent);
selectNum = find(selectScore>Thr,1);

% newFeature = score(:,1:selectNum);
processedData = (Data - mu)*coeff(:,1:selectNum);
PCAProcessor.mu = mu;
PCAProcessor.coeff = coeff;
PCAProcessor.score = score;
PCAProcessor.latent = latent;
PCAProcessor.selectNum = selectNum;
PCAProcessor.selectScore = selectScore;
end