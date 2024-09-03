function overallPerf = getOverallPerformance(loss,Nc,razorK)
% calculate overall performance
overallPerf = 1./(loss) .* (1 - razorK) + razorK.* 1./log10(Nc);
end