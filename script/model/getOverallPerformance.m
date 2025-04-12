function [overallPerf,perfL,perfC] = getOverallPerformance(loss,Nc,razorK)
% calculate overall performance
perfL = 1./(loss);
perfC = 1./(log10(Nc));
overallPerf = perfL .* (1 - razorK) + razorK.* perfC;
end