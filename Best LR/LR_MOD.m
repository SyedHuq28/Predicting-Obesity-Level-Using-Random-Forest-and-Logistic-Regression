%Ref- https://uk.mathworks.com/help/stats/mnrval.html

clear all

% Load Train and Test Data
traindata = readtable('train_mod.csv');
testdata = readtable('test_mod.csv');

% Convert to array format which can be later helpful to do matrix
% calculation
X = table2array(traindata(:,1:11));
Y = table2array(traindata(:,12));
x = table2array(testdata(:,1:11));
y = table2array(testdata(:,12));
Y = Y + 1;
y = y + 1;

% Fitting the model
B = mnrfit(X,Y,'Model','hierarchical');
yhat = mnrval(B,x);
[val, index] = max(yhat, [], 2);

% Accuracy of the model, acc
acc = mean(y == index);
a=confusionmat(y,index);

cmt = a';%Transpose
diagonal = diag(cmt);
sumr = sum(cmt,2);

%Finding precision
precision = diagonal ./ sumr;
overall_prec = mean(precision);

%Finding recall
sumc = sum(cmt,1);
recall = diagonal ./ sumc;
overall_re = mean(recall);

%Finding f1_score
f1_score = ((overall_prec*overall_re)/(overall_prec+overall_re));