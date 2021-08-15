Window = 500;
rng('default')
trainFlag = randperm(size(Feature_Input_split_train,2),62432);
testFlag = randperm(size(Feature_Input_split_test,2),31216);

Input_train_split_data = Feature_Input_split_train(:,trainFlag);
Target_train_split_data = Target_train_split_data(:,trainFlag);
Input_test_split_data = Feature_Input_split_test(:,testFlag);
Target_test_split_data = Target_test_split_data(:,testFlag);

XTrain = reshape(Input_train_split_data,[2,2,1,size(Input_train_split_data,2)]);
XTest = reshape(Input_test_split_data,[2,2,1,size(Input_test_split_data,2)]);

YTrain = zeros(size(Target_train_split_data,2),1);
for i = 1:size(Target_train_split_data,2)
    YTrain(i) = find(Target_train_split_data(:,i) == 1);
end
YTrain = categorical(YTrain);

YTest = zeros(size(Target_test_split_data,2),1);
for i = 1:size(Target_test_split_data,2)
    YTest(i) = find(Target_test_split_data(:,i) == 1);
end
YTest = categorical(YTest);

imageSize = [2,2,1];

layers = [
    imageInputLayer(imageSize)

    convolution2dLayer([2 2],50,'padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([2 2],100,'padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([2 2],200,'padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([2 2],100,'padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([2 2],50)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    %dropoutLayer(0.1)
    %fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer]


options = trainingOptions('adam',...
    'MaxEpochs',30, ...
    'MiniBatchSize', 256, ...
    'Verbose',false,...
    'Plots','training-progress');     

cnn = trainNetwork(XTrain,YTrain,layers,options);
%cnn = trainNetwork(XTrain,YTrain,layers,options);
predictedLabels_train = classify(cnn,XTrain);
valLabels_train = YTrain;
accuracy_train_cnn = sum(predictedLabels_train == valLabels_train)/numel(valLabels_train)

predictedLabels = classify(cnn,XTest);                                              
valLabels = YTest;
accuracy_test_cnn = sum(predictedLabels == valLabels)/numel(valLabels)

Input_test_split_data = Feature_Input_split_test;
Target_test_split_data = Target_test_split_data;

XTest = reshape(Input_test_split_data,[2,2,1,size(Input_test_split_data,2)]);

Target_test_split_data = [HC_test_split_target I_test_split_target L_test_split_target M_test_split_target ...
R_test_split_target TI_test_split_target TL_test_split_target TM_test_split_target TR_test_split_target TT_test_split_target];
YTest = zeros(size(Target_test_split_data,2),1);
for i = 1:size(Target_test_split_data,2)
    YTest(i) = find(Target_test_split_data(:,i) == 1);
end
YTest = categorical(YTest);

predictedLabels = classify(cnn,XTest);                                              
valLabels = YTest;
accuracy_test_cnn = sum(predictedLabels == valLabels)/numel(valLabels)

data = double(predictedLabels);

number = 160;
data_target = zeros(number,1);
for i = 1:size(data_target,1)
data_target(i) = find(Target_test_raw_data(:,i) == 1);
end

b1 = Target_test_split_data;
d = zeros(size(data,1),1);
 for i = 1:size(data,1)
     d(i) = find(b1(:,i) == max(b1(:,i)));
 end
 
c = data;

c1 = zeros(size(data,1),1);
 for i = 1:number
     c1((i-1)*1951+1:i*1951) = mode(c((i-1)*1951+1:i*1951));
 end
 
temp = zeros(number,1);
 for i = 1:number
     temp(i) = mode(c((i-1)*1951+1:i*1951));
 end
 
 accuracy = size(find(c == d),1)/size(data,1)
 accuracy1 = size(find(c1 == d),1)/size(data,1)   
 accuracy2 = size(find(temp == data_target),1)/size(data_target,1)

