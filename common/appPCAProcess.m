
function [processedData] = appPCAProcess(Data,PCAProcess)
processedData = (Data - PCAProcess.mu)*PCAProcess.coeff(:,1:PCAProcess.selectNum);
end
