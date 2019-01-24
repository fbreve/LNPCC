% This is an example of the Particle Competition and Cooperation for 
% Semi-Supervised Learning with Label Noise method applied to the Wine Data
% Set from the UCI Machine Learning Repository
% Available at: https://archive.ics.uci.edu/ml/datasets/wine
%
% by Fabricio Breve - 24/01/2019
%
% Loading the Wine Data Set
load wine.data
% Getting the dataset attributes (all colums, except the first one).
X = wine(:,2:end);
% Getting dataset labels (first column). Labels should be >0 and in
% sequence. Ex.: 1, 2, 3.
label = wine(:,1);
% Randomly selecting 10% of the labels to be presented to the algorithm.
% 40% of the labeled examples will have their label changed (label noise)
% An unlabeled item is represented by 0.
slabel = slabelgenwl(label,0.1,0.4);
disp('Parameters k: 10, distance: Normalized Euclidean, others: Default.');
disp('Running the algorithm on the Wine Dat Set with 10% labeled examples, from which 40% have a wrong label (label noise)...');
tStart = tic;
owner = lnpcc(X, slabel, 10, 'seuclidean');
tElapsed = toc(tStart);
% Evaluating the classification accuracy.
acc = stmwevalk(label,slabel,owner);
fprintf('Classification accuracy: %0.4f - Execution Time: %0.4fs\n\n',acc,tElapsed);