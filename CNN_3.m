Window = 500;
rng('default')
trainFlag = randperm(size(Input_train_split_data,2),62432);
%testFlag = randperm(size(Input_test_split_data,2),31216);
% % n = size(Input_train_split_data,2);
% % % n1 = n*(3/4);
% % random_0 = randperm(n);
% % random = random_0;
% % % random = random_0(1:n1);
Input_train_split_data = Input_train_split_data(trainFlag);
Target_train_split_data = Target_train_split_data(:,trainFlag);
% % 
% % m = size(Input_test_split_data,2);
% % %m1 = m*(3/4);
% % random_0_1 = randperm(m);
% % random_1 = random_0_1;
% % %random_1 = random_0_1(1:m1);
Input_test_split_data = Input_test_split_data;
Target_test_split_data = Target_test_split_data;


XTrain = zeros(Window,2,1,numel(Input_train_split_data));
a = cell2mat(Input_train_split_data); 
b = zeros(Window,2,1,numel(Input_train_split_data));
for i = 1:numel(Input_train_split_data)
    b(:,:,1,i) = a(:,((i-1)*2+1):i*2);
end
XTrain = b;
a = [];
b = [];


XTest = zeros(Window,2,1,numel(Input_test_split_data));
a = cell2mat(Input_test_split_data); 
b = zeros(Window,2,1,numel(Input_test_split_data));
for i = 1:numel(Input_test_split_data)
    b(:,:,1,i) = a(:,((i-1)*2+1):i*2);
end
XTest = b;


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

imageSize = size(XTrain);

imageSize = imageSize(1:3);

layers = [
    imageInputLayer(imageSize)

    convolution2dLayer([2 2],50)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    convolution2dLayer([10 1],50)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 1],'Stride',[2,1])
   

    convolution2dLayer([10 1],100)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    convolution2dLayer([10 1],30)
    batchNormalizationLayer
    reluLayer
  
    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    fullyConnectedLayer(10)
    %dropoutLayer(0.1)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer]


options = trainingOptions('adam',...
    'MaxEpochs',20, ...
    'Verbose',false,...
    'Plots','training-progress'); 
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',5, ...
    

%imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
%ds = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);

cnn = trainNetwork(XTrain,YTrain,layers,options);
%cnn = trainNetwork(XTrain,YTrain,layers,options);
predictedLabels_train = classify(cnn,XTrain);
valLabels_train = YTrain;
accuracy_train_cnn = sum(predictedLabels_train == valLabels_train)/numel(valLabels_train)

predictedLabels = classify(cnn,XTest);                                              
valLabels = YTest;
accuracy_test_cnn = sum(predictedLabels == valLabels)/numel(valLabels)

target_test = zeros(10,size(YTest,1));
for i = 1:size(YTest,1)
    target_test(YTest(i,:),i) = 1;
end

y = grp2idx(predictedLabels);
output_test = zeros(10,size(y,1));
for i = 1:size(y,1)
    output_test(y(i,:),i) = 1;
end
y=[];

plotconfusion(target_test,output_test)

target_train = zeros(10,size(YTrain,1));
for i = 1:size(YTrain,1)
    target_train(YTrain(i,:),i) = 1;
end

y = grp2idx(predictedLabels_train);
output_train = zeros(10,size(y,1));
for i = 1:size(y,1)
    output_train(y(i,:),i) = 1;
end

plotconfusion(target_train,output_train)

