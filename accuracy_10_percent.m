%data = readNPY('new_result_10_12_2018.npy');
data = double(predictedLabels);
data = y';
%data = y1';
%number = 20;
number = 160;
data_target = zeros(number,1);
for i = 1:size(data_target,1)
data_target(i) = find(Target_test_raw_data(:,i) == 1);
end

%Test_Output
b1 = Target_test_split_data;
d = zeros(size(data,1),1);
 for i = 1:size(data,1)
     d(i) = find(b1(:,i) == max(b1(:,i)));
 end
 
c = zeros(size(data,1),1);
for i = 1:size(data,1)
     c(i) = find(data(i,:) == max(data(i,:)));
end
%c = data;

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
 
temp = temp';
data_target = data_target';
[c_matrix,Result,RefereceResult]= confusion.getMatrix(data_target,temp)
a = [RefereceResult.Sensitivity RefereceResult.Specificity RefereceResult.Precision RefereceResult.F1_score RefereceResult.MatthewsCorrelationCoefficient];