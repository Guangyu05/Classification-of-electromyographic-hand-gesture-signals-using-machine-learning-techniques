Feature_Input_split_train = zeros(4,size(Input_train_split_data,2));

for i = 1:size(Input_train_split_data,2)
    Feature_Input_split_train(1,i) = sum(abs(Input_train_split_data{i}(:,1)));
    Feature_Input_split_train(2,i) = sum(abs(Input_train_split_data{i}(:,2)));
    Feature_Input_split_train(3,i) = std(abs(fft(Input_train_split_data{i}(:,1))));
    Feature_Input_split_train(4,i) = std(abs(fft(Input_train_split_data{i}(:,2))));
end


Feature_Input_split_test = zeros(4,size(Input_test_split_data,2));

for i = 1:size(Input_test_split_data,2)
    Feature_Input_split_test(1,i) = sum(abs(Input_test_split_data{i}(:,1)));
    Feature_Input_split_test(2,i) = sum(abs(Input_test_split_data{i}(:,2)));
    Feature_Input_split_test(3,i) = std(abs(fft(Input_test_split_data{i}(:,1))));
    Feature_Input_split_test(4,i) = std(abs(fft(Input_test_split_data{i}(:,2))));
end