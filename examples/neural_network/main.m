train_X=readmatrix("train_X.txt",  'Delimiter',',')';
train_y=readmatrix("train_y.txt",  'Delimiter',',')';
test_X=readmatrix("test_X.txt",  'Delimiter',',')';
test_y=readmatrix("test_y.txt",  'Delimiter',',')';
train_data=[train_X,train_y];
test_data=[test_X,test_y];
data=[train_data;test_data];
sqrt(sum(test_y.^2)/length(test_y))

writematrix(data,"data.csv");