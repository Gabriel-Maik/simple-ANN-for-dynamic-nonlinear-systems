function [InputMat, OutputMat] = Signals2LearningData(Input, Output, InputsNum, IsComplex)
% Converts 1D signals into matrices appriopriate for ANN learning
%
% Input     N x 1 vector real/complex
% Output    N x 1 vector real/complex
% InputsNum 1 x 1 scalar integer
% IsComplex 1 x 1 scalar boolean
%

N = length(Input);
if IsComplex
  InputMat = zeros(InputsNum*2,N);
  OutputMat = zeros(2,N);
  OutputMat(1,:) = real(Output);
  OutputMat(2,:) = imag(Output);
  for Idx = 1:(InputsNum-1)
    InputMat(2*(1:Idx)-1,Idx) = real(Input(Idx:-1:1));
    InputMat(2*(1:Idx),Idx) = imag(Input(Idx:-1:1));
  end
  for Idx = InputsNum:N
    InputMat(1:2:end-1,Idx) = real(Input(Idx:-1:Idx-InputsNum+1));
    InputMat(2:2:end,Idx) = imag(Input(Idx:-1:Idx-InputsNum+1));
  end
else
  if ~isreal(Input) || ~isreal(Output)
    error('Data is complex!');
  end
  InputMat = zeros(InputsNum,N);
  OutputMat = Output';
  for Idx = 1:(InputsNum-1)
    InputMat(1:Idx,Idx) = Input(Idx:-1:1);
  end
  for Idx = InputsNum:N
    InputMat(:,Idx) = Input(Idx:-1:Idx-InputsNum+1);
  end
end

end