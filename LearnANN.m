function [ModelANN] = LearnANN(InputMat, OutputMat, IsComplex, ParamNum, LayerNum, EpochsNum, BatchSize)
% Learns ANN model based on properly arranged data matrices and other parameters
%
% InputMat    ? x N   matrix  real/complex
% OutputMat 1/2 x N   matrix  real/complex
% IsComplex   1 x 1   scalar  boolean
% ParamNum    1 x 1   scalar  integer       Approximate number of model parameters
% LayerNum    1 x 1   scalar  integer       Number of layers
% EpochsNum   1 x 1   scalar  integer       Number of epochs
%

InputMatSize = size(InputMat);
InputsNumAll = InputMatSize(1); 

NeuronsPerLayer = round(ParamNum/LayerNum);

% Learning settings

options = trainingOptions("adam", ...
    ...'ExecutionEnvironment', 'gpu',... % Uncomment to enable GPU processing (was slower last time)
    InitialLearnRate=3e-4, ...
    SquaredGradientDecayFactor=0.99, ...
    MaxEpochs=EpochsNum, ...
    MiniBatchSize=BatchSize, ...
    Plots="training-progress");

% Alternative method
% TrainingMethod = 'trainscg';
% NeuronsInHiddenLayer = ones(1,LayerNum)*NeuronsPerLayer;
% Creating Artificial Neural Network
% ModelANN = feedforwardnet(NeuronsInHiddenLayer, TrainingMethod);
% 
% ModelANN.divideParam.trainRatio = 90/100;
% ModelANN.divideParam.valRatio = 10/100;
% ModelANN.divideParam.testRatio = 0/100;
% 
% ModelANN.trainParam.epochs = EpochsNum;
% ModelANN.trainParam.goal = 0.0;
% ModelANN.trainParam.show = 10;
% ModelANN.trainParam.max_fail = ModelANN.trainParam.epochs;
% [ModelANN,trening] = train(ModelANN,InputMat,OutputMat,'Options',options);

HiddenLayers = [fullyConnectedLayer(NeuronsPerLayer,'Name','conn_1')...
    tanhLayer('Name','tansig_1')];
for Idx = 2:LayerNum
  HiddenLayers = [HiddenLayers fullyConnectedLayer(NeuronsPerLayer,'Name', ...
    strcat('conn_',num2str(Idx)) )...
    tanhLayer('Name', strcat('tansig_',num2str(Idx)) )];
end

OutputLayerSize = 1;
if IsComplex == true
  OutputLayerSize = 2*OutputLayerSize;
end
Layers = [...
  featureInputLayer(InputsNumAll,'Name','input')...
  HiddenLayers...
  fullyConnectedLayer(OutputLayerSize,'Name','conn_output')...
  regressionLayer('Name','output')];

lgraph = layerGraph(Layers);

[ModelANN,TrainingInfo] = trainNetwork(InputMat.', OutputMat.', lgraph, options);
TrainingInfo

end