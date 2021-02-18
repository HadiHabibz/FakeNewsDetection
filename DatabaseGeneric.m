classdef ( Abstract ) DatabaseGeneric 
    
    properties( Access = protected )
        
        % 'Rawdata' is a cell with N rows and 20 columns, where N is the
        % number of valid entries in the dataset. Each column is a string.
        rawData;
               
        % This is the unique features (words) in the dataset.
        % This does not include the corresponding frequencies.
        % We use the same features for each category, which should 
        % not be a big deal if we select the top K Most repeated words.        
        features;
        
        % This holds the corresponding labels for each entry in rawData
        % For training datasets, these labels are used for training and
        % For test dataset, these labels are used for evaluation
        labels;
        
        % Number of classes, e.g., if it is binary classification
        % then this is set to 2
        numberOfClasses;
        
        % Prior probabilities of each class or categories. This is a vector
        % with the length 'numberOfClasses'
        priorProbabilities;
        
        % All parameters we need for estimating Pd. The size and structure
        % of this parameter changes based on the ddistribution we use for
        % vector P. For example, MDD uses D+1 features, so this will be a
        % vector of length D+1. 
        likelihoodParameters;
        
        % Datasets used for training and testing. Generate these using
        % split function.
        trainingDataset;
        testDataset;
        
    end % class data members
    
    methods( Access = public )
        
        % Constructor
        function obj = DatabaseGeneric( pathToDataset )
            
            obj.rawData = 0;
            obj.numberOfClasses = 2;
            obj.trainingDataset = 0;
            obj.testDataset = 0;                       
            
            % Basically, there are two constructors. 
            % One loads the data from a file (i.e., dataset)
            % The other uses an existing dataset to create another
            % one. This ones acts more or less like copy-constructor.
            % The option 'do not load' separates the two.
            if ~strcmp( pathToDataset, 'do not load' )
                obj = loadAllData( obj, pathToDataset );                  
            end % end if                
                        
        end % function constructor

        % Train classifier
        % User can determine the type of distribution they want to use for
        % estimating the probabilities of the multinomial distribution
        % using the 'distribution' parameter. The valid arguemtns are 
        % 'multinomial - naive', 'MDD', 'MGDD', and MBLM, which
        % respectively refer to using frequencies (See Eq. 3), Multinomial
        % Dirichlet distribution (See Eq. 5), Generalized Multinomial
        % Dirichlet distribution (See Eq. 8) and Multinomial Beta-Lioville
        % Distribution (See Eq. 24). 
        function obj = trainClassifier( obj, distribution, initialvalues )
            
            if nargin < 3
                initialvalues = [];
            end % if
            
            % Compute all unique words in the training dataset. This will
            % constitute all features. This function uses a hyper parameter
            % N, which is the number of features. For example, if N = 100,
            % then the function selects the top 100 most repeated words.
            % It is better to select N words for each category separately,
            % but I select the same features of all datasets. If N is
            % relatively small (e.g., less than 1000) this should not make
            % a huge difference. I only do this because I was lazy! If you
            % have tim, customize features for each category. However, some
            % changes might be required in the code.
            obj.trainingDataset = computeAllUniqueWords(...
                obj.trainingDataset );
            
            % Extract features of the training dataset. 
            % This is just the labels of the features (words, if the
            % dataset is a document) and does not include the corresponding
            % frequency. 
            obj.features = getFeatures( obj.trainingDataset );
            
            % Prior probabilities must be set based on the relative 
            % population of each category. E.g., if 80% of the dataset
            % is class 0 while the rest 20% is class 1, then priors are set
            % to 0.8 and 0.2, respectively.
%             obj.priorProbabilities = ...
%                 computePriorProbabilities( obj.trainingDataset );
            
            % Determine the type of classifiers (e.g., MDD, MGDD, etc.)
            mode = translateClassifierType( obj, distribution );
            
            % Mode 0 corresponds to Eq. 3. This method does not use any
            % parameters and hence no optimization is required. All other
            % distributions require optimization.
            if mode == 0    
                
                obj.likelihoodParameters = ...
                    computeMultinomialPrameters( obj.trainingDataset );
                
                return;
                
            end % if
            
            % Optimzie the parameters of the selected distribution using
            % gradient descent. Because our parameters are mostly
            % uncostrained, we can use gradient descent.
            obj.likelihoodParameters = ...
                    computeParameters_GradientDescent(...
                    obj.trainingDataset, mode, initialvalues );
                                            
        end % function trainClassifier        
        
        function obj = clasifyDataset( obj, classifierType )
           
           modeStrings = { 'Naive', 'Naive-GD', 'MDD', 'MGDD', 'MBLM' };
           mode = translateClassifierType( obj, classifierType );
           mode = mode + 1;
                     
           [scores, dataLabels, probabilities] = computeScores( obj,...
               obj.trainingDataset, classifierType );      
           
           save( ['Results/Training_scores_',...
               modeStrings{mode}], 'scores' );
           
           save( ['Results/Training_labels_',...
               modeStrings{mode}], 'dataLabels' );
           
           save( ['Results/Training_probabilities_',...
                modeStrings{mode}], 'probabilities' );
           
           [scores, dataLabels, probabilities] = computeScores( obj,...
               obj.testDataset, classifierType );  
           
           save(['Results/Test_scores_',...
               modeStrings{mode}], 'scores' );
           
           save( ['Results/Test_labels_',...
               modeStrings{mode}], 'dataLabels' );     
           
           save( ['Results/Test_probabilities_',...
                modeStrings{mode}], 'probabilities' );
           
        end % function clasifyDataset
        
        % Determine what type of classifier that is requested. This enables
        % us to make customization depending on the distribution. Putting 
        % everything in one function causes consistency. These values for
        % mode are used throughout the code. It is recommended not to
        % change them.
        function mode = translateClassifierType( obj, classifierType )
            
            % Use exactly these strings
            if strcmp( classifierType, 'multinomial - naive' )
                mode = 0;
                
            elseif strcmp( classifierType,...
                    'multinomial - gradient descent' )
                mode = 1;
                
            elseif strcmp( classifierType, 'MDD' )
                mode = 2;
                
            elseif strcmp( classifierType, 'MGDD' )
                mode = 3;
                
            elseif  strcmp( classifierType, 'MBLM' )
                mode = 4;
                
            else
                fprintf( ['The selected distribution is invalid',...
                    'Use:\n1. multinomial - naive\n2. MDD\n',...
                    '3. MGDD\n', '4. MBLM\n'] );
                exit();
                
            end % if..else                
            
        end % function translateClassifierType
               
        % Split the dataset into two parts
        % The first part is used to training
        % The second part is used for testing.
        % The size of the training dataset is 
        % controlled by 'trainingToAllRatio'.
        % When set to (say) 0.9, the training dataset
        % will have a size 90% of the complete
        % dataset and the test dataset size will be
        % 10% of the complete dataset. The elements
        % in each dataset are determined randomly.
        function obj = ...
                splitDatasetRandomly(...
                obj,...
                trainingToAllRatio )
            
            % Keep this for regenerating results
            rng( 0 );
            
            % Give a random sequence from 1 to N, where numberOfEntries
            % is the number of entries in the dataset. This is also 
            % the first dimension of the obj.rawData (excluding) the
            % header.
            numberOfEntries = size( obj.rawData, 1 ) - 1;
            shuffledRows = randperm( numberOfEntries );
            
            % Pick the first trainingToAllRatio*100 percent of
            % elements for the training dataset and the rest for
            % test
            trainingSetSize = trainingToAllRatio * numberOfEntries;
            trainingSetSize = floor( trainingSetSize );
            
            % Pick the first 'trainingSetSize' random indices for training
            % dataset and the rest for test dataset
            entriesForTraining = shuffledRows( 1:trainingSetSize );
            selectedForTest = shuffledRows( trainingSetSize+1:end );
            
            % Pick the entries corresponding to the selected random indices
            rawDataTrain = obj.rawData( entriesForTraining, : );
            rawDataTest = obj.rawData( selectedForTest, : );
            
            % Create new objects.
            % Option '1' instruct to the function to extract features
            obj.trainingDataset = createNewDataset(...
                obj, rawDataTrain, obj.numberOfClasses,...
                obj.labels( entriesForTraining ), 1 );
            
            % There is no need to wordify the test dataset
            % Option 0 instruct the function not to extract features. We
            % will extract them entry by entry during classification. There
            % is no need to re-do this. (This procedure can be slow for
            % large documents!)
            obj.testDataset = createNewDataset(...
                obj, rawDataTest, obj.numberOfClasses,...
                obj.labels( selectedForTest ), 0  );     
            
        end % function splitDataset     
        
        function parameters = getLikelihoodParameters( obj )
            
            parameters = obj.likelihoodParameters;
            
        end % function getLikelihoodParameters
        
        % Getter function for rawData
        function returnedRawData = getRawData( obj )
            
            returnedRawData = obj.rawData;
            
        end % function getRawData
        
        % Getter function for label at a given index
        function returnedLabel = getLabel( obj, index )
            
            returnedLabel = obj.labels( index );
            
        end % function getLabel
        
    end % class public services
    
    methods( Access = protected )
        
        % Compute the likelihood of all entries of the current dataset.
        % If there are N entries in the dataset, the scores will be an
        % N by M matrix, where M is the number of classes. To classify each
        % entry, we pick the hypothesis corresponding to the maximum score.
        % Do not forget that these values must be multiplied by the prior 
        % before making the decisionm unless the prior is uniform. The 
        % second output, 'labels' returns the corresponding correct label
        % of each entry. This is not the label we guess but instead the 
        % ground truth. This makes it easier to verifty the classification 
        % result later on.
        function [scores, labels, averageProbabilities] =...
                computeScores( ...
                obj, datasetToClassify, classifierType )  
                    
            % Extract the raw data to see how many entries there are
            rawDataToClassify = getRawData( datasetToClassify );
            
            % Pre-allocate memory for scores and labels. Scores is N by M
            % where N is the number of entries (data samples) in the
            % dataset, while M is the number of classes. 'labels' is N by
            % 1.
            scores = zeros( size( rawDataToClassify, 1 ),...
                obj.numberOfClasses );
            
            averageProbabilities = zeros( ...
                size( rawDataToClassify, 1 ), obj.numberOfClasses,...
                length( obj.features ) );
            
            labels = zeros(  size( rawDataToClassify, 1 ), 1 ); 
            
            % Determine which distribution we are working with
            % I should've made each distribution a separate class!
            % Didn't think it through well.
            mode = translateClassifierType( obj, classifierType );
                                              
            % Classification will take a while
            % Do it in parallel. 
            parfor entry = 1:size( rawDataToClassify, 1 )                           
                
                % Create a single entry dataset with current entry
                % Make sure to extract features (by using option '1')
                relevantDataset = createNewDataset( obj, ...
                    rawDataToClassify( entry, : ),...
                    obj.numberOfClasses,...
                    getLabel( datasetToClassify, entry ), 1 );    
                
                % Compute the scores for this single entry dataset
                % This function is the log of Eq. 2. At this point, Pd is
                % already estimated.
                [scores( entry, : ), averageProbabilities( entry, :, : )] =...
                    computeLogLikelihood( obj, relevantDataset, mode );
                
                % Extract the corresponding label.
                labels( entry ) = getLabel( relevantDataset, 1 );
                                              
            end % for entry
            
        end % function comptueScores
        
        % Compute log of likelihood
        % This is the log of Eq. 2, but because different distributions use
        % different approaches to estimate Pd, we must invoke the right
        % function. This function is a gateway to the right function.
        function [scores, probabilities] = ...
                computeLogLikelihood( obj, relevantDataset, mode )
            
            % initialize estimation parameters based on the type of 
            % the selected distribution .
            [alpha, beta] = initializeParameters( obj, mode );
            
            % Compute the scores for this single entry dataset                            
            if mode == 0 || mode == 1
                [scores, probabilities] = ...
                    getMultinomialLikelihoodForEntry( ...
                    obj, relevantDataset );

            elseif mode == 2                    
                [scores, probabilities] = ...
                    getMultinomialLikelihoodForEntry_MDD( ...
                    obj, alpha, relevantDataset );

            elseif mode == 3
                [scores, probabilities] = ...
                    getMultinomialLikelihoodForEntry_MGDD( ...
                    obj, alpha, beta, relevantDataset );

            else 
                [scores, probabilities] = ...
                    getMultinomialLikelihoodForEntry_MBLM( ...
                    obj, relevantDataset );
                
            end % if..else
            
        end % function computeLogLikelihood
        
        % Intitialize parameters that we use in our estimation.
        % Because each distribution uses different number of parameters, we
        % need to see these accordingly.
        function [alpha, beta] = initializeParameters( obj, mode )
            
            % These values will be re-written if mode ~= 0. 
            % We have initialize them here though or else PARFOR will get
            % confused! 
            alpha = 0;
            beta = 0;
            
            % Extract parameters that we use for each distribution.
            % If the selected distribution is MDD (corresponding to mode 2
            % ) there is only alpha, which is a vector of length D+1, where
            % D is the number of features.
            if mode == 2
                alpha = obj.likelihoodParameters;                
            end % if
            
            % If mode is three, then the selected distribution is MGDD,
            % which includes alpha and beta as parameters, both of which
            % are vectors with size D, where D is the number of features
            if mode == 3
                alpha = squeeze( obj.likelihoodParameters(...
                    :, 1:length( obj.features )-1 ) );
                beta = squeeze( obj.likelihoodParameters(...
                    :, length( obj.features ):end ) );                
            end % if
            
        end % function initializeParameters
        
        % Compute the log likelihood of Eq. 2, given the MDD is selected.
        % First, compute Pd's, after that the problem reduces to computing
        % the log-likelihood of mutlinomial (See. Eq. 2).
        function [logLikelihood, probabilities] = ...
                getMultinomialLikelihoodForEntry_MDD(...
                obj, alpha, singleEntryDataset )         
            
            % These correspond to Pd's, in Eq. 2
            % Note that each class has its own Pd's
             probabilities = zeros( obj.numberOfClasses,...
                 length( obj.features ) );
             
             % Data corresponds to Xid's in Eq. 2
             % These are the number of times that each feature is repeated.
             data = getFeatureFrequency(...
                 singleEntryDataset, obj.features );
             
             % Use Eq. 5 to estimate probabilities
             for classIndex = 1:obj.numberOfClasses
                 
                 % Use Eq. 5 to compute probabilities
                 probabilities( classIndex, : ) = ...
                     convertParameterToProbability_MDD(...
                     obj, data, alpha( classIndex, : )' );                        
                                  
             end % for classIndex             
             
             % From now, it is like multinomial
             obj.likelihoodParameters = probabilities;
             logLikelihood = getMultinomialLikelihoodForEntry(...
                 obj, singleEntryDataset );
                        
        end % function getMultinomialLikelihoodForEntry_MDD
        
        function probabilities = ...
                convertParameterToProbability_MDD(...
                obj, data, alpha )
            
             probabilities = ( alpha + data ) ./...
                 ( sum( alpha ) + sum( data ) ); 
            
        end % function convertParameterToProbability_MDD
        
        function probabilities = ...
                convertParameterToProbability_MGDD(...
                obj, data, alpha, beta, partialSums )
            
             data = data( 1:end-1 );
             probabilities = log ( alpha + data ) -...
                 log( alpha + beta + partialSums( 1:end-1 ) );
             
             term1 = zeros( length( alpha ), 1 );
             
             for i = 2:length( alpha )
                 term1( i ) =...
                     sum( log( beta( 1:i-1 ) + partialSums( 2:i ) ) -...
                 log( alpha( 1:i-1 ) + beta( 1:i-1 ) +...
                 partialSums( 1:i-1 ) ) );
             end % for i
                          
             probabilities = exp( probabilities + term1 );
             probabilities( end + 1 ) = 1 - sum( probabilities );
             
             % A hack to prevent p = 0
             % This will cause problems with log(p) later on            
             if probabilities( end ) <= 0
                 [maxProb, maxProbLocation] = max( probabilities );
                 p = probabilities( maxProbLocation ) * (1-1e-6);
                 probabilities( maxProbLocation ) = p;
                 probabilities( end ) = 1e-6 * maxProb;
             end % if
            
        end % function convertParameterToProbability_MGDD
        
        function probabilities = ...
            convertParameterToProbability_MBLM(...
            obj, data, alpha, ALPHA, beta )
            
             dataLast = data( end );
             data = data( 1:end-1 );             
             dataSum = sum( data );
             
             probabilities = ...
                 ( ALPHA + dataSum ) ./ ...
                 ( ALPHA + dataSum + beta + dataLast );
             
             probabilities = probabilities .* ...
                 ( alpha + data ) / ( sum( alpha + data ) );
                          
             probabilities( end + 1 ) = 1 - sum( probabilities );
            
        end % function convertParameterToProbability_MBLM
        
        % Compute log-likelihood if MGDD is selected by the user.
        % This distribution uses two parameter vectors alpha and beta
        % 'alphaComplete' and 'betaComplete' are 2D matrices that include
        % parameters for all classes.
        function [logLikelihood, probabilities] = ...
                getMultinomialLikelihoodForEntry_MGDD(...
                obj, alphaComplete, betaComplete, singleEntryDataset )         
            
             % This corresponds to Xid in Eq. (8)
             data = getFeatureFrequency(...
                 singleEntryDataset, obj.features );       
             
             obj.likelihoodParameters = zeros(...
                 obj.numberOfClasses, size( alphaComplete, 2 )+1 );
             
             % This corresponds to Nil
             partialSums = zeros( length( data ), 1 );             
             partialSums( 1 ) = sum( data );             
             
             for i = 2:length( partialSums )
                
                 % Though confusing, this is the same as
                 % partialSums( i ) = sum( data( i:end ) );
                 % but more efficient
                 partialSums( i ) = partialSums( i-1 ) - data( i-1 );                
               
             end % for i                      

             % Compute Pd's for Eq. 2 for each class 
             for classIndex = 1:obj.numberOfClasses
                                   
                 % Extract probabilties corresponding to the current class.
                 alpha = alphaComplete( classIndex, : )';
                 beta = betaComplete( classIndex, : )';
                 
                 probabilities = convertParameterToProbability_MGDD(...
                    obj, data, alpha, beta, partialSums );
                 
                 % Compute PD+1 where D+1 is the number of features
                 probabilities( end ) = 1 - sum( probabilities( 1:end-1 ) );
                 
                 % Set parameters
                 obj.likelihoodParameters( classIndex, : ) = probabilities;
                         
             end % for classIndex      
             
             probabilities = obj.likelihoodParameters;
             
             logLikelihood = getMultinomialLikelihoodForEntry(...
                 obj, singleEntryDataset );
                        
        end % function getMultinomialLikelihoodForEntry_MGDD
        
        function [logLikelihood, probabilities] = ...
                getMultinomialLikelihoodForEntry_MBLM(...
                obj, singleEntryDataset )         
            
             probabilities = zeros( obj.numberOfClasses,...
                 length( obj.features ) );
             
             data = getFeatureFrequency(...
                 singleEntryDataset, obj.features );

             for classIndex = 1:obj.numberOfClasses
                 
                 alpha = obj.likelihoodParameters( classIndex, 1:end-2 )';
                 ALPHA = obj.likelihoodParameters( classIndex, end-1 );
                 beta = obj.likelihoodParameters( classIndex, end );
                 
                 probabilities( classIndex, 1:end-1 ) = ...
                     ( ( ALPHA + sum( data( 1:end-1 ) ) ) /...
                     ( ALPHA + sum( data ) + beta ) ) * ...
                     ( ( alpha + data( 1:end-1 ) ) ./...
                     sum( data( 1:end-1 ) + alpha ) );
                 
                 probabilities( classIndex, end ) = 1 - ...
                     sum( probabilities( classIndex, 1:end-1 ) );
                 
             end % for classIndex             
             
             obj.likelihoodParameters = probabilities;
             logLikelihood = getMultinomialLikelihoodForEntry(...
                 obj, singleEntryDataset );
                        
        end % function getMultinomialLikelihoodForEntry_MBLM
        
        
        % Given a single entry dataset, compute its likelihood as explained
        % int Eq. (3) of the reference. the output will be a signle
        % integer. Note that we are computing the log of Eq. (3). 
        function [logLikelihood, probabilities] = ...
                getMultinomialLikelihoodForEntry(...
                obj, singleEntryDataset )         
            
             % This corresponds to xId in Eq. (3)
             data = getFeatureFrequency( singleEntryDataset, obj.features ); 
             logLikelihood = zeros( obj.numberOfClasses, 1 );            
              
             for classIndex = 1:obj.numberOfClasses
                 
                 % Extract parameters corresponding to this class. For
                 % multinomial, parameters are Pd's that we learn during
                 % the training phase.
                 probabilities = obj.likelihoodParameters( classIndex, : );   
                 
                 % Hopefully, this should never not happen. 
                 if any( probabilities == 0 )
                     logLikelihood = 0;
                     return;
                 end % if
                 
                 logLikelihood( classIndex ) = ...
                     getMultinomialProbability(...
                     obj, probabilities', data );
                                  
             end % for classIndex             
                        
        end % function getMultinomialLikelihoodForEntry
        
        function p = getMultinomialProbability( obj, probabilities, data )
            
             % There will be lots of zero features
             % There are the words that encountered in the training dataset
             % but do not exist in the current entry. As seen in Eq. 3 the
             % contribution of these entries will be zero to log
             % likelihood, so they can be ignored to save some computation
             % and avoid strange cases such as log( 0 ).
             nonZeroLocations = ( data ~= 0 );
             
             % Sum data once to save some computation
             allDataSum = sum( data );
             
             term1 = sum( data( nonZeroLocations ) .*...
                     log( probabilities( nonZeroLocations ) ) );
                 
             % This is the Sterling's approximation for log( data! )
             % We must handel cases, where data is zero
             term2 = data( nonZeroLocations ) .*...
                 log( data( nonZeroLocations ) ) - ...
                 data( nonZeroLocations );
             term2 = -1 * sum( term2 );
             
             % Sterling's approximation for log[ (sum( data ))! ]         
             term3 = allDataSum * log( allDataSum ) - allDataSum;
             
             p = ( term1 + term2 + term3 );
            
        end % getMultinomialProbability
                
        % Optimize the 'parameters' using gradient descent. Notice that
        % gradient descent is an unconstraint optimization. The input
        % 'mode' determines the gradient vector that we compute, which
        % changes based on type of classifier that we have used. In any
        % case, the cost function that try to minimize is the difference
        % between log of the posteriro probability of non-target data and
        % target data (sum_over_all_training_dataset: logP(nonTarget) - 
        % logP(target)).
        function parameters = ...
                computeParameters_GradientDescent( ...
                obj, mode, initialvalues )
            
            % Mode determines the type of classifier and distribution we
            % are using. By default it is set to multinomial
            if nargin < 2
                mode = 0;
            end % if
            
            classCount = obj.numberOfClasses;
            obj.priorProbabilities =...
                computePriorProbabilities( obj );
            
            % Gamma is the step size
            gamma = 250;            
%             gamma = 2;
            trialsCount = 600;
            
            if isempty( initialvalues )
                parametersToOptimize = ...
                    getParametersToOptimizeForGD( obj,...
                    length( obj.features ), mode );     
            else 
                parametersToOptimize = initialvalues;
                
            end % if..else
            
            datasetToClassify = obj;            
            rawDataToClassify = getRawData( datasetToClassify ); 
            
            gradient = zeros( [size( rawDataToClassify, 1 ),...
                size( parametersToOptimize )] );                         
            
            selectedCategories = zeros( size( rawDataToClassify, 1 ), 1 );
            scores = zeros( size( rawDataToClassify, 1 ), 1 );
            allScores = zeros( trialsCount, 3 ); 

            for trial = 1:trialsCount
                
                parfor entry = 1:size( rawDataToClassify, 1 )
%                 for entry = 1:size( rawDataToClassify, 1 )

                    H = zeros( classCount, 1 );
                    Z = zeros( classCount, 1 );

                    singleEntryDataset = createNewDataset( obj, ...
                        rawDataToClassify( entry, : ),...
                        obj.numberOfClasses,...
                        getLabel( datasetToClassify, entry ), 1 ); 

                    data = getFeatureFrequency(...
                        singleEntryDataset, obj.features );
                    
                    nonZero = ~( data == 0 );
                    
                    if ~any( data )
                        continue;
                    end
                   
                    
                    currentGradient = ...
                        zeros( size( parametersToOptimize ) ); 

                    for hypothesis = 1:classCount

                        [currentGradient( hypothesis, : ),...
                            H( hypothesis ),...
                            Z( hypothesis )] = ...
                            computeGradient( obj,...
                            datasetToClassify, data,...
                            parametersToOptimize( hypothesis, : )',...
                            entry, hypothesis, mode ); 
                        
                        H( hypothesis ) = H( hypothesis ) + ...
                            log( obj.priorProbabilities( hypothesis ) ) - ...
                            sum( data( nonZero ) .*...
                            log( data( nonZero ) ) - data( nonZero ) ) + ...
                            sum( data ) * log( sum( data ) ) - ...
                            sum( data );
                        
                        currentGradient( hypothesis, : ) = ...
                            -currentGradient( hypothesis, : ) ./...
                            H( hypothesis );  
                        
                    end % for hypothesis
                    
                    % [~, selectedCategories( entry )] = max( Z );                    
                    selectedCategories( entry ) =...
                        getLabel( datasetToClassify, entry )+1;
                    gradient( entry, : ) = currentGradient( : ); 
                    scores( entry ) = -log( sum( 1.001.^( H ) ) );

                end % for entry
                
                selectedCategories = selectedCategories - 1;
                allGradients = zeros( size( gradient, 2 ),...
                    size( gradient, 3 ) );
                
                for i = 1:obj.numberOfClasses
                    inCategory = ( selectedCategories == i - 1 );                    
                	allGradients( i, : ) = squeeze( ...
                        sum( gradient( inCategory, i, : ), 1 ) ); 
                    
                    if sum(  inCategory ) == 0
                        allGradients( i, : ) = 0;
                    else 
                        allGradients( i, : ) = ...
                            allGradients( i, : ) / sum( inCategory );
                    end
                end % for i
                
                 [norm( allGradients( 1, : ) ),...
                     norm( allGradients( 2, : ) )]
                 sum( scores )
                 
                allScores( trial, 1 ) = sum( scores );
                allScores( trial, 2 ) = norm( allGradients( 1, : ) );
                allScores( trial, 3 ) =  norm( allGradients( 2, : ) );
                
                unconstrained = parametersToOptimize -...
                    gamma * allGradients;
                violations = ( unconstrained(:) <= 0 );
                allGradients( violations ) = 0;                      

                parametersToOptimize = ...
                    parametersToOptimize -...
                    gamma * allGradients;

            end % trial

            save( ['loglikelihood_', 'MODE_', num2str( mode ), '_T'...
                num2str( trialsCount ), '_', num2str( gamma )] );
            parameters = parametersToOptimize;
                    
        end % function computeParameters_GradientDescent
       
        function [gradient, H, categoryP] = computeGradient( obj,...
                datasetToClassify,...
                data,...
                parametersToOptimize,...
                entry,...
                category,...
                mode )
            
            H = 0;
            categoryP = 0;
            
            if mode == 1                             
                gradient = ...
                    computeGradient_Multinomial( obj,...
                    datasetToClassify, data,...
                    parametersToOptimize, entry, category );  

            elseif mode == 2
                [gradient, H, categoryP] = ...
                    computeGradient_MDD( obj,...
                    datasetToClassify, data,...
                    parametersToOptimize, entry, category );  
                
            elseif mode == 3
                [gradient, H, categoryP] = ...
                    computeGradient_MGDD( obj,...
                    datasetToClassify, data,...
                    parametersToOptimize( 1:length( obj.features )-1 ),...
                    parametersToOptimize( length( obj.features ):end ),...
                    entry, category );
                
            elseif mode == 4
                [gradient, H, categoryP] = ...
                    computeGradient_MBLM( obj,...
                    datasetToClassify, data,...
                    parametersToOptimize,...                    
                    entry, category );
                
            end % if
            
        end % function computeGradient
       
        function parameters =...
                getParametersToOptimizeForGD(...
                obj, featuresCount, mode )
                        
            if mode == 3
                % rng( 2 );
                parameters = ones( obj.numberOfClasses,...
                    2 * ( featuresCount - 1 ) );                  

            elseif mode == 4
                rng( 5 );
                parameters = ones( obj.numberOfClasses,...
                    featuresCount + 1, 1 );

            else
                % rng( 1 );
                parameters = ones( obj.numberOfClasses, featuresCount );                

            end % if..else
                                     
        end % function getParametersToOptimizeForGD
        
        % Compute gradient of multinomial distribution
        % Using gradient descent on multimodial distribution is wrong must
        % be done at user's risk. The problem is that gradient descent
        % uses unconstraint optimization. But the parameters in multinomial
        % are constraint to add up to 1
        function gradients = computeGradient_Multinomial( obj,...
                datasetToClassify, data, parametersToOptimize,...
                entry, category )
            
            % Taking the partial derivative of log of Eq.2 for Pd, yields a
            % very simple gradient 
            gradients = data ./ parametersToOptimize;
            
            % Our cost function is the sum of the posterior probability on
            % non-target data minus the sum of posterior of target data,
            % which we aim to minimize
            if getLabel( datasetToClassify, entry ) == category-1
                gradients = -1 * gradients;
            end % if
            
        end % function computeGradient_Multinomial
        
        function [gradients, H, categoryProbability] =...
                computeGradient_MDD( obj,...
                datasetToClassify, data, parametersToOptimize,...
                entry, category )
                                 
            % Here, alpha is alpha j and data is data i
            % Look at my notes
            alpha = parametersToOptimize;
            
            probability = convertParameterToProbability_MDD(...
                obj, data, alpha );
            
            categoryProbability = obj.priorProbabilities( category ) *...
                getMultinomialProbability( obj, probability, data );
            
            gradients = data .* ( 1 ./ ( data + alpha ) -...
                sum( data ) ./ ( sum( alpha ) + sum( data ) ) ); 
            
%             H = data .* ( log( data + alpha ) -...
%                 log( sum( alpha ) + sum( data ) ) );
            
            H = sum( data .* log( probability ) );
            
        end % function computeGradient_MDD
        
        function [gradients, H, categoryProbability] =...
                computeGradient_MGDD( obj,...
                datasetToClassify, data, alpha, beta,...
                entry, category )
           
            partialSums = zeros( length( alpha )+1, 1 ); 
            partialSums( 1 ) = sum( data( 1:end ) );   
                                   
            for i = 2:length( partialSums )                
                partialSums( i ) = partialSums( i - 1 ) - data( i - 1 );               
            end % for i
            
            term1 = alpha + beta;
            term2 = term1 + partialSums( 1:end-1 );
            
            probability = convertParameterToProbability_MGDD(...
                obj, data, alpha, beta, partialSums );    

            categoryProbability = obj.priorProbabilities( category ) *...
                getMultinomialProbability( obj, probability, data );
            
            dataFull = data;
            data = data( 1:end-1 );
            
            partialAlpha = data ./ ( alpha + data );
            partialAlpha = partialAlpha - data ./ term2;
            partialAlpha = partialAlpha - partialSums( 2:end ) ./ term2;
            
            partialBeta = - data ./ term2;
            partialBeta = partialBeta + partialSums( 2:end ) ./...
                ( beta + partialSums( 2:end ) );
            partialBeta = partialBeta - partialSums( 2:end ) ./ ...
                ( term2 );  
            
            H = dataFull .* log( probability );       
            H = sum( H );
            gradients = [partialAlpha, partialBeta];
            gradients = gradients(:);
            
            if any( ismissing( gradients ) )
                a = 2;
            end
            
        end % function computeGradient_MGDD
        
        function [gradients, H, categoryProbability] = ...
                computeGradient_MBLM( obj,...
                datasetToClassify, data, parametersToOptimize,...
                entry, category )
           
            alpha = parametersToOptimize( 1:end-2 );
            ALPHA = parametersToOptimize( end-1 );
            beta = parametersToOptimize( end );           
            dataSum = sum( data ); 
            dataSumIncomplete = dataSum - data( end );
            
            probability = convertParameterToProbability_MBLM(...
                obj, data, alpha, ALPHA, beta );    

            categoryProbability = obj.priorProbabilities( category ) *...
                getMultinomialProbability( obj, probability, data );
            
            partialAlpha = ( data( 1:end-1 ) ./ ...
                ( alpha + data( 1:end-1 ) ) ) - ...
                dataSum / ( dataSumIncomplete + sum( alpha ) );
            
            partialALPHA = dataSum / ( ALPHA + dataSumIncomplete ) - ...
                dataSum / ( ALPHA + dataSum + beta );
            
            partialBeta = -dataSum / ( ALPHA + dataSum + beta );     
            H = sum( data .* log( probability ) );           
            gradients = [partialAlpha', partialALPHA, partialBeta]';

            
        end % function computeGradient_MBLM
        
        
        % Compute parameters that we need for calculating multinomial
        % likelihodd (See Eq. 2 of Reference 1). We need N parameters,
        % in total, where N is the number of features. We compute them
        % by dividing the frequency of each word by the total number of 
        % words in the training dataset
        function parameters = computeMultinomialPrameters( obj )
            
            % Parametes is a 2D matrix. There is a row for each category.
            % That is if the classification is binary, there will be two
            % rows only. Each row includes N parameters, where N is the
            % number of features.
            parameters = zeros( obj.numberOfClasses,...
                length( obj.features ) );
            
            % For each class, compute the parameters using the training
            % dataset
            for category = 1:obj.numberOfClasses
                
                % Pick all entries that belong to this class and put them
                % in a separate dataset. This enables us to reuse many
                % features of the class
                entiresInCategory = ( obj.labels == category-1 );
                rawDataInCategory = obj.rawData( entiresInCategory, : );
                relevantDataset = createNewDataset( obj, ...
                    rawDataInCategory, obj.numberOfClasses,...
                    obj.labels( entiresInCategory ), 1 );
                
                wordFrequency = zeros( length( obj.features ), 1 );
                
                % For each attribute, compute the number of times it is
                % encountered in this category
                for attribute = 1:length( obj.features )
                    
                     wordFrequency( attribute ) = ...
                        getFeatureFrequency(...
                        relevantDataset, obj.features( attribute ) );
                                        
                end % for attribute
                
                % Divide by total to convert to probabilities. 
                % Ensure it adds up to 1.
                violations = ( wordFrequency == 0 );   
                
                wordFrequency( 1:end ) = wordFrequency( 1:end ) /...
                    sum( wordFrequency( 1:end ) );
                
                wordFrequency( end ) = 1 - sum( wordFrequency( 1:end-1 ) );  
                
                if sum( violations ) ~= 0
                    [minVal, minLocation] = ...
                        min( wordFrequency( ~violations ) );                
                    wordFrequency( ~violations( minLocation ) ) = ...
                        0.99 * minVal;                
                    wordFrequency( violations ) = ( 0.01 * minVal ) ./...
                        sum( violations );
                end % if
                
                parameters( category, : ) = wordFrequency;
                
            end % for category
            
        end % function computeMultinomialPrameters
        
        % Compute the prior probabilities
        % To this end, look at the dataset and use the relative frequency
        % of each label as its corresponding prior probability
        function p = computePriorProbabilities( obj )
            
            % Pre-allocate memory (Good practice)
            p = zeros( obj.numberOfClasses, 1 );
            
            % Obtain the number of times each label is repeated in the
            % dataset
            for i = 1:obj.numberOfClasses
                
                p( i ) = sum( obj.labels == i-1 );
                
            end % for i
            
            % Convert the counts or frequencies to ratios
            p = p / length( obj.labels );
            
        end % function computePriorProbabilities
        
        % A setter function for rawData. Used when we clone things
        function obj = setRawData( obj, newRawData )

            obj.rawData = newRawData;

        end % function setRawData 
        
        % setter function for parameters
        function obj = setParameters( obj, parameters )
            
            obj.likelihoodParameters = parameters;
            
        end % function setParameters
        
        % A setter function for number of classes.
        function obj = setNumberOfClasses( obj, classesCount )
            
            obj.numberOfClasses = classesCount;
            
        end % function setNumberOfClasses
        
        % A setter function for labels
        function obj = setLabels( obj, newLabels )
            
            obj.labels = newLabels;
            
        end % function setLabels
               
        % Getter function for features
        function returnedFeatures = getFeatures( obj )
            
            returnedFeatures = obj.features;
            
        end % function getFeatures
                
    end % class protected utilities
    
    % Abstract services     
    methods( Abstract, Access = protected )
        
        obj = loadAllData( obj, pathToData );
        features = computeAllUniqueWords( obj );        
        featuresCount = getFeatureFrequency( obj, feature );
        newDataset = createNewDataset( obj, newRawData, ...
            numberOfClasses, labels, optionalFlags );
        
    end % class public abstract services
    
end % class DatabaseGeneric