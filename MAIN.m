% Main script

%N = 1000;
%S = 1;
MemoryLen = S + R;
ParamNum = 1000; % 1000
LayersNum = 5;
EpochsNum = 2; % 10
IsComplex = false;
BatchSize = 2048; % mini-batch size 64 is default /4 for smallest number of data, /64 for long sim.

% u = complex(rand(N,1),rand(N,1));
% u = rand(N,1);
% y = conv(u,(S+1):-1:1,'full');
% y = y(1:N);
% y = y.^2;

[InputMat, OutputMat] = Signals2LearningData(u(1:N),y(1:N),MemoryLen+1,IsComplex);
ModelANN = LearnANN(InputMat,OutputMat,IsComplex,ParamNum,LayersNum,EpochsNum,BatchSize);

% u2 = complex(rand(N,1),rand(N,1));
% u2 = rand(N,1);
% y2 = conv(u2,(S+1):-1:1,'full');
% y2 = y2(1:N);
% y2 = y2.^2;

[InputMat2, OutputMat2] = Signals2LearningData(u_test(1:N),y_test(1:N),MemoryLen+1,IsComplex);
ANNModelPrediction = PredictANN(ModelANN,InputMat2,IsComplex);

if ~exist('ShowPlots','var') || ShowPlots
  plot(real([w_test ANNModelPrediction]));
end
errorANN = rmse(w_test(1:N), ANNModelPrediction);