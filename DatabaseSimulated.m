classdef DatabaseSimulated < DatabaseGeneric
    
    properties( Access = private )
        
        featuresCount;
        entriesCount;
        newsLength;
        
    end % class properties 
    
    methods( Access = public )
        
        function obj = DatabaseSimulated( pathToDataset )
            
            % Call parent's constructor
            obj@DatabaseGeneric( pathToDataset );
            
        end % constructor
                
        function obj = setFeatures( obj, newFeatures )
            
            obj.features = newFeatures;
            
        end % function setFeatures
        
        function obj = setFeaturesCount( obj, newFeatureCount )
            
            obj.featuresCount = newFeatureCount;
            
        end % function setFeaturesCount
                
    end % class public methods
    
    methods( Access = protected )
        
        function obj = loadAllData( obj, pathToData )
            
            obj.featuresCount = 100;
            obj.entriesCount = 10*1000;
            obj.newsLength = 1000;
            
            obj.rawData = zeros( obj.entriesCount, obj.featuresCount );
            obj.labels = zeros( obj.entriesCount, 1 );
            obj.features = 1:obj.featuresCount;
                        
            for i = 1:obj.entriesCount
                
                randomLabel = randi( obj.numberOfClasses ) - 1;
                randomLength = getARandomLength( obj );
                
                if randomLabel == 0 
                    selectedWords = rand( randomLength, 1 );
                    
                else
                    selectedWords = getRandomNormalInRange(...
                        obj, randomLabel/obj.numberOfClasses,...
                        randomLabel/( 2 * obj.numberOfClasses ),...
                        randomLength );
                    
                end % if
                
                selectedWords = selectedWords * obj.featuresCount;
                selectedWords = ceil( selectedWords );
                
                for j = 1:length( selectedWords )
                    obj.rawData( i, selectedWords( j ) ) = ...
                        obj.rawData( i, selectedWords( j ) ) + 1;                 
                end % for j
                
                obj.labels( i ) = randomLabel;
                
            end % for i
                       
        end % function loadDataset
        
        % Given newRawData, create a new dataset that uses this new raw
        % data. Make sure to pass 'do not load' as it instruct the
        % constructor not to load data from secondary memory.
        function newDataset = createNewDataset(...
                obj, newRawData, classesCount, newLabels, wordifyFlag )
            
            if nargin <= 4
                wordifyFlag = 0;
            end
            
            newDataset = DatabaseSimulated( 'do not load' ); 
            newDataset = setRawData( newDataset, newRawData );     
            newDataset = setNumberOfClasses( newDataset, classesCount );
            newDataset = setLabels( newDataset, newLabels );
            newDataset = setFeaturesCount( newDataset, obj.featuresCount );
            
            newDataset = setParameters(...
                newDataset, obj.likelihoodParameters );        
            
        end % function createNewDataset

        
        % Given a feature, determine how many times it is repeated in the
        % dataset. 
        function featuresCount = getFeatureFrequency( obj, feature )
            
            featuresCount = zeros( length( feature ), 1 );
            
            for i = 1:length( feature )
                
                currentFeature = sum( obj.rawData( :, feature( i ) ) );
                featuresCount( i ) = currentFeature;

            end % for i
            
        end % function getFeatureFrequency
        
        % Determine the number of unique words in the dataset
        function obj = computeAllUniqueWords( obj )
            
            obj.features = 1:obj.featuresCount;
            
        end % function computeAllUniqueWords
        
        function randomNumber = ...
                getRandomNormalInRange( obj, mu, sigma, count )
            
            randomNumber = normrnd( mu, sigma, count, 1 );
            
            for i = 1:length( randomNumber )
                
                while randomNumber( i ) <= 0 || randomNumber( i ) >= 1
                    randomNumber( i ) = normrnd( mu, sigma );
                end
                
            end % for i
            
        end % getRandomNormalInRange
        
        function randomLength = getARandomLength( obj )
            
            randomLength = 0;
            
            while randomLength <= 0
                randomLength = normrnd( obj.newsLength, obj.newsLength/10 );
                randomLength = round( randomLength );
            end % while 
            
            
        end % function getARandomLength
        
    end % methdos
    
end % class DatabaseSimulated