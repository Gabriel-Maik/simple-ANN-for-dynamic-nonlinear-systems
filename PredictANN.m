function ModelOutput = PredictANN(ModelANN, InputMat, IsComplex)
% Predicts the output based on the ANN model
%
% ModelANN    1 x 1   object
% InputMat    ? x N   matrix  real/complex
% OutputMat 1/2 x N   matrix  real/complex
%

ModelOutputMat = predict(ModelANN, InputMat.');
ModelOutputMat = ModelOutputMat.';

if IsComplex
  ModelOutput = ModelOutputMat(1,:).' + 1i*ModelOutputMat(2,:).';
else
  ModelOutput = ModelOutputMat(1,:).';
end

end