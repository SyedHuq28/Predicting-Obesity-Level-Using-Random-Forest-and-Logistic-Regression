%I did a course which helped to learn and get some reference- https://www.udemy.com/course/machine-learning-for-datascience-using-matlab

clc
clear all
close all
warning off
%Read Data
data = readtable('mod_data4.csv');

%Create the model with the target variable 'result'
mod1 = fitcensemble(data,'result','Method','AdaBoostM2', 'NumLearningCycles',462,'Learners','tree');

%Do partitioning of testing and training data with KFold CV 
mod2 = cvpartition(mod1.NumObservations, 'KFold', 5);

%Fit and predict the model
final_mod = crossval(mod1,'cvpartition',mod2); 
'HoldOut', 0.2
% predict according to the number of folds
pred_K_1 = predict(final_mod.Trained{1},data(test(mod2,1),1:end-1));
pred_K_2 = predict(final_mod.Trained{2},data(test(mod2,2),1:end-1));
pred_K_3 = predict(final_mod.Trained{3},data(test(mod2,3),1:end-1));
pred_K_4 = predict(final_mod.Trained{4},data(test(mod2,4),1:end-1));
pred_K_5 = predict(final_mod.Trained{5},data(test(mod2,5),1:end-1));

% results of the predictions
r1 = confusionmat(final_mod.Y(test(mod2,1)),pred_K_1);
r2 = confusionmat(final_mod.Y(test(mod2,2)),pred_K_2);
r3 = confusionmat(final_mod.Y(test(mod2,3)),pred_K_3);
r4 = confusionmat(final_mod.Y(test(mod2,4)),pred_K_4);
r5 = confusionmat(final_mod.Y(test(mod2,5)),pred_K_5);

% a is the Confusion matrix which is the combined result
a = r1+r2+r3+r4+r5;


x = sum(diag(a)); %Finding diagonal of the matrix
a1 = sum(a(1,:));
b1 = sum(a(2,:));
b2 = sum(a(3,:));
b3 = sum(a(4,:));
b4 = sum(a(5,:));
b5 = sum(a(6,:));
b6 = sum(a(7,:));

%Using the formula sum of diagonal/sum of all other places to find the accuracy
acc = (x/(a1+b1+b2+b3+b4+b5+b6))*100 


%Finding other scores

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
f1_score = (2*(overall_prec*overall_re)/(overall_prec+overall_re));


