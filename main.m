% simulatedDataset = DatabaseSimulated( 'blank' );
% myDatabaseSplit = splitDatasetRandomly( simulatedDataset, 0.9 );
% 
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'multinomial - naive' );
% clasifyDataset( myDataset_AfterTraining, 'multinomial - naive' );
% 
% tic
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'MDD'  );
% clasifyDataset( myDataset_AfterTraining, 'MDD' );
% toc
% save( 'myDataset_AfterTraining', 'myDataset_AfterTraining' );

% tic
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'MGDD'  );
% clasifyDataset( myDataset_AfterTraining, 'MGDD' );
% toc
% save( 'myDataset_AfterTraining', 'myDataset_AfterTraining' );

% tic
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'MBLM'  );
% clasifyDataset( myDataset_AfterTraining, 'MBLM' );
% toc
% save( 'myDataset_AfterTraining', 'myDataset_AfterTraining' );

%% Real Data
% myDatabase = Database( 'fake.csv' );
% myDatabaseSplit = splitDatasetRandomly( myDatabase, 0.7 );
% myDatabaseSplit = load( 'myDatabaseSplit.mat' );
% myDatabaseSplit = myDatabaseSplit.myDatabaseSplit;

% tic
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'multinomial - naive' );
% clasifyDataset( myDataset_AfterTraining, 'multinomial - naive' );
% toc

% tic
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'MDD'  );
% clasifyDataset( myDataset_AfterTraining, 'MDD' );
%  save( 'myDataset_AfterTraining1', 'myDataset_AfterTraining' );
% toc

% tic
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'MGDD'  );
% clasifyDataset( myDataset_AfterTraining, 'MGDD' );
% toc

% tic
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'MBLM'  );
% clasifyDataset( myDataset_AfterTraining, 'MBLM' );
%  save( 'myDataset_AfterTraining2', 'myDataset_AfterTraining' );
% toc

% tic
% myDataset_AfterTraining = ...
%     load( 'Results\RealData\myDataset_AfterTraining_MBLM_2048_G2' );
% myDataset_AfterTraining = myDataset_AfterTraining.myDataset_AfterTraining;
% myDataset_AfterTraining = trainClassifier(...
%     myDatabaseSplit, 'MBLM',...
%     getLikelihoodParameters( myDataset_AfterTraining )  );
% clasifyDataset( myDataset_AfterTraining, 'MBLM' );
%  save( 'myDataset_AfterTraining2', 'myDataset_AfterTraining' );
% toc