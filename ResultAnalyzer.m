classdef ResultAnalyzer
    
    properties( Access = private )
        
        linewidth;
        fontsize;
        
    end % properties
    
    methods( Access = public )
        
        function obj = ResultAnalyzer()
            
            obj.linewidth = 4;
            obj.fontsize = 32;
                       
        end % constructor
        
        function [data, labels, logLikelihoods] = ...
                loadResults( obj, classifierType, testOrTrain )
            
            logLikelihoods = 0;
            
            data = load( ['Results\', testOrTrain, '_scores_',...
                classifierType, '.mat'] );
            
            labels = load( ['Results\', testOrTrain, '_labels_',...
                classifierType, '.mat'] );
            
            if strcmp( testOrTrain, 'Training' )
                logLikelihoods = load( ['Results\', 'logLikelihood_',...
                    classifierType, '.mat'] );
            end % if
            
            data = data.scores;
            labels = labels.dataLabels; 
                       
        end % function loadResults
        
        function plotAll( obj, distributionType )
            
            [scoresTrain, labelsTrain, logLikelihood] = ...
                loadResults( obj, distributionType, 'Training' );            
            
            title = ['Discriminating Training Data - ', distributionType] ;
            plotNormalDistribution( obj, scoresTrain, labelsTrain, title );   
%             plotGradientAndScore( obj, logLikelihood, distributionType );
            
            [scoresTest, labelsTest] = ...
                loadResults( obj, distributionType, 'Test' );            
            title = ['Discriminating Test Data - ', distributionType] ;
            plotNormalDistribution( obj, scoresTest, labelsTest, title );
%             plotClassAccuracy( obj, scoresTest, labelsTest );
            
        end % function plotAll
        
        function plotAverageLearnedProbabilities( obj )
            
            methods = {'MDD', 'MGDD', 'MBLM'};
            
            for k = 1:length( methods )
                
                probabilitiesTraining = ...
                    load( ['Results\Trials_4096_Gamma_2\',...
                    'Training_probabilities_', methods{k}] );

                probabilitiesTraining =...
                    probabilitiesTraining.probabilities;

                labels = load( 'Results\Trials_4096_Gamma_2\Training_labels_MBLM' );
                labels = labels.dataLabels; 

                averageProb0 = zeros( size( probabilitiesTraining, 3 ), 1 );
                averageProb1 = zeros( size( probabilitiesTraining, 3 ), 1 );

                for i = 1:size( probabilitiesTraining, 1 )

                    if labels( i ) == 0
                        averageProb0 = averageProb0 +...
                            squeeze( probabilitiesTraining( i, 1, : ) );

                    else 
                        averageProb1 = averageProb1 +...
                            squeeze( probabilitiesTraining( i, 2, : ) );

                    end % if

                end % for i

                averageProb0 = averageProb0 / sum( labels == 0 );
                averageProb1 = averageProb1 / sum( labels == 1 );

                figure;
                bar( averageProb0 );
                title( 'Cat 0' );
                xlabel( 'Word ID' );
                ylabel( 'Probability' );

                figure;
                bar( averageProb1 );
                title( 'Cat 1' );
                xlabel( 'Word ID' );
                ylabel( 'Probability' );
            
            end % for k 
            
        end % function plotAverageLearnedProbabilities
        
        function plotExampleNews( obj, dataset )
            
            rawData = getRawData( dataset );
            
            for label = 1:2
                
                while true

                    randomEntry = randi( size( rawData, 1 ), 1, 1 );
                    entryLabel = getLabel( dataset, randomEntry );

                    if entryLabel == label-1 
                        break;
                    end % if

                end % while
                
                figure;
                data = rawData( randomEntry, : ); 
                bar( data );
                ylabel( 'Count' );
                xlabel( 'Word ID' );
                            
            end % for
            
            
            
            
        end % function plotExampleNews
        
        function plotGradientAndScore( obj, loglikelihood, type )
            
            
            loglikelihood = loglikelihood.allScores;
            goal = loglikelihood( :, 1 );
            figure;
            plot( 1:length( goal ), goal/abs( max( goal ) ), 'LineWidth', obj.linewidth );
            xlabel( 'Iteration (\times10^3)' );
            ylabel( 'Unnormalized Likelihood' );
            title( ['Total Log Likelihood - ', type] );
            set( gca, 'FontSize', obj.fontsize );
            xlim( [1, length( goal )] );
            xticks( 0:1000:length( goal ) );
            xticklabels( ( 0:1000:length( goal )) / 1000 );
            grid ON;
            
            derivatives = zeros( size( loglikelihood, 1 ), 1 );
            
            for i = 1:size( loglikelihood, 1 )
                derivatives( i ) = norm( loglikelihood( i, 2:end ) );
            end % for i     
            
            figure;
            plot( derivatives, 'LineWidth', obj.linewidth );
            xlabel( 'Iteration (\times10^3)' );
            ylabel( 'Derivative Norm' );
            title( [' Derivative Norm - ', type] );
            set( gca, 'FontSize', obj.fontsize );
            xlim( [1, size( loglikelihood, 1 )] );
           xticks( 0:1000:length( goal ) );
            xticklabels( ( 0:1000:length( goal )) / 1000 );
            grid ON;
            
        end % function
        
        function data = getDataInClass( obj, scores, labels, class )
                                   
            inClassIndex = ( labels == class );
            data = scores( inClassIndex, : );
            data = data(:);
            violations = ( data == -Inf ) & ( data == Inf );          
            
            if any( sum( violations ) )
                fprintf( 'Warning: INF scores!\n' );
            end % if
               
            data = data( ~violations );                                          
            violations = ismissing( data );
               
            if sum( violations ) ~= 0
                fprintf( 'Warning: %d missing scores\n',...
                    sum( violations ) );
            end % if     
            
            data = data( ~violations ); 
            data = reshape( data, [], size( scores, 2 ) );
           
        end % function getDataInClass
        
        function plotClassAccuracy( obj, scores, labels )
            
            accuracies = computeClassAccuracy( obj, scores, labels );
            
            figure;
            histogram( accuracies );
            set( gca, 'FontSize', obj.fontsize );
            
        end % function plotClassAccuracy
                        
        function accuracies = computeClassAccuracy( obj, scores, labels )
            
            classesCount = size( scores, 2 );
            accuracies = zeros( classesCount, 1 );
            
            for i = 1:classesCount
                
                data = getDataInClass( obj, scores, labels, i-1 );                
                                
                correctDetection = 0;
                
                for k = 1:length( data )
                    
                    odds = zeros( size( data, 2 ), 1 );
                    
                    for j = 1:size( data, 2 )
                        odds( j ) = data( k, j ) / ...
                            ( sum( data( k, : ) ) - data( k, j ) );
                    end % for j
                    
                    [~, guessedClass] = max( odds );
                    
                    if guessedClass == i
                        correctDetection = correctDetection + 1;
                    end % if                    
                    
                end % for k
                
                accuracies( i ) = correctDetection / length( data );
                
            end % for i
            
        end % function computeClassAccuracy
        
        function plotNormalDistribution( obj, scores, labels, plotTitle )
            
            classesCount = size( scores, 2 );                    
            standardDeviations = zeros( classesCount, 1 );
            means = zeros( classesCount, 1 );
            
           for i = 1:classesCount
               
               data = getDataInClass( obj, scores, labels, i-1 );               
               standardDeviations( i ) = std( data( :, 1 ) ./  data( :, 2 ) );
               means( i ) = mean( data( :, 1 ) ./  data( :, 2 )  );
               
           end % for i
           
           distanceFromMean = 4 * max( standardDeviations );
           plotRange = linspace( min( means ) - distanceFromMean,...
               max( means ) + distanceFromMean, 10*1000 );
           
           figure;
           legendText = {};
           for i = 1:classesCount
               
               y = normpdf( plotRange, means( i ),...
                   standardDeviations( i ) );
               y = y / max( y );
               
               plot( plotRange, y, 'LineWidth', obj.linewidth );    
               legendText( i ) = {['Class ', int2str(i)]};
               hold on;
               
           end % for i
           
           xlabel( ' Log-Posterior ' );
           ylabel( 'Distribution' );       
           title( plotTitle );
           myLegend = legend( legendText );
           set( gca, 'FontSize', obj.fontsize );
           grid ON;
           hold off;
            
        end % function plotNormalDistribution
        
    end % class public services
    
end % class ResultAnalyzer