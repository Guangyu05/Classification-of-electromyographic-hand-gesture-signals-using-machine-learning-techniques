Window = 500;
trainFlag = randperm(size(Input_train_split_data,2),62432);
Input_train_split_data = Input_train_split_data(trainFlag);
Target_train_split_data = Target_train_split_data(:,trainFlag);

XTrain = zeros(Window,2,numel(Input_train_split_data));
a = cell2mat(Input_train_split_data); 
b = zeros(Window,2,numel(Input_train_split_data));
for i = 1:numel(Input_train_split_data)
    b(:,:,i) = a(:,((i-1)*2+1):i*2);
end
XTrain = b;
a = [];
b = [];
x = reshape(XTrain,[1000,numel(Input_train_split_data)]);
t = Target_train_split_data;

XTest = zeros(Window,2,numel(Input_test_split_data));
a = cell2mat(Input_test_split_data); 
b = zeros(Window,2,numel(Input_test_split_data));
for i = 1:numel(Input_test_split_data)
    b(:,:,i) = a(:,((i-1)*2+1):i*2);
end
XTest = b;
a = [];
b = [];
x1 = reshape(XTest,[1000,numel(Input_test_split_data)]);
t1 = Target_test_split_data;


rng('default')
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network

hiddenLayerSize = [21,21,21]
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error
net.trainParam.epochs = 5000;

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,x,t,'useParallel','yes','showResources','yes');
% [net,tr] = train(net,x,t); 
% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end

trainingaccuracy=trainingresult(net,x,t)
y_train = net(x);
foo1 = zeros(size(y_train,2),1);
 for i = 1:size(y_train,2)
     foo1(i) = find(y_train(:,i) == max(y_train(:,i)));
 end
 
 foo2 = zeros(size(t,2),1);
 for i = 1:size(t,2)
     foo2(i) = find(t(:,i) == max(t(:,i)));
 end
 
 train_accuracy = size(find(foo1 == foo2),1)/size(foo1,1)   

testingaccuracy=testresult(net,x1,t1)
%testingaccuracy=testresult(net,Feature_Input_split_test,Target_test_split_data)

function[accuracy]=trainingresult(network,trainginput,trainingoutput)
% Get the output with training input.
result=sim(network,trainginput);
% The position of maximum value among outputs represents the predicted
% label of input features.
predict_output=zeros(10,size(trainginput,2));
for j=1:size(trainginput,2)
A=result(:,j) ;
[M,L] = max(A(:));
predict_output(L,j)=1;
end
% Count the correctly predicted samples.
sample=0;
total=size(trainginput,2);
for j=1:size(trainginput,2)
if predict_output(:,j)==trainingoutput(:,j)
sample=sample+1;
end
end
% Calculate the average accuracy of training samples.
accuracy=sample/total;
end
function[accuracy]=testresult(network,testinginput,testingoutput)
% Get the output with testing input.
result=sim(network,testinginput);
predict_output=zeros(10,size(testinginput,2));
for j=1:size(testinginput,2)
A=result(:,j) ;
[M,L] = max(A(:));
% The position of maximum value among outputs represents the predicted
% label of input features.
predict_output(L,j)=1;
end
sample=0;
total=size(testinginput,2);
for j=1:size(testinginput,2)
if predict_output(:,j)==testingoutput(:,j)
sample=sample+1;
end
end
% Calculate the average accuracy of testing samples.
accuracy=sample/total;
% Plot confusion matrix with testing accuracy of each class.
%plotconfusion(testingoutput,predict_output);
end

target = zeros(size(t,2),1);
for i = 1:size(target,1)
target(i) = find(t(:,i) == 1);
end
k = 1;
KNNnet = fitcknn(x',target,'NumNeighbors',k,'Standardize',1);
y = zeros(size(t1,2),1);
tic
for i = 300001:312160
    y(i) = predict(KNNnet,x1(:,i)');
end
toc