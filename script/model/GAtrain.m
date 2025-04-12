function model = GAtrain(model,loss,nvp)

arguments
    model
    nvp.CausalMask (1,1) logical = true
    nvp.MutatuionProb = 0.05;
    nvp.Dropout (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(nvp.Dropout,1)} = 0
    nvp.InputMask = []
end

end