classdef Database < DatabaseGeneric
    
    properties( Access = private )       
                
        % this is a hash table or an array that uses words as keys and
        % return their corresponding frequency
        words;
        
        dataField;
        
    end % properties
    
    methods( Access = public )
        
        % Constructor
        function obj = Database( pathToDataset )
            
            % Call parent's constructor
            obj@DatabaseGeneric( pathToDataset );
                       
            % Create a blank hash table
            % Given a word as a string, this map returns its frequency
            obj.words = containers.Map(...
                'KeyType', 'char', 'ValueType', 'double' );
            
            obj.numberOfClasses = 2;
            obj.dataField = 6;
                                   
        end % constructor function  
        
    end % class public services
                             
    methods( Access = protected )
        
        % Determine the number of unique words in the dataset
        function obj = computeAllUniqueWords( obj )
            
            numberOfWords = 100;
            allValues = cell2mat( values( obj.words ) );
            featuresComplete = keys( obj.words );  
            obj.features = cell( numberOfWords, 1 );        
            
            while numberOfWords >= 1
                
                [~, maxLocation] = max( allValues );
                obj.features( numberOfWords ) = ...
                    featuresComplete( maxLocation );
                allValues( maxLocation ) = -1;
                numberOfWords = numberOfWords - 1;
                
            end % while
            
            obj.features = obj.features( 1:end-5 );
            
        end % function computeAllUniqueWords
        
        % Given a feature, determine how many times it is repeated in the
        % dataset. We scale everything by a factor of 1/10 to make the
        % numbers more managable
        function featuresCount = getFeatureFrequency( obj, feature )
            
            featuresCount = zeros( length( feature ), 1 );
            %obj = computeAllUniqueWords( obj );
            
            for i = 1:length( feature )
                
                currentFeature = feature{i};

                % If the key is not encountered before, then its frequency
                % should be set to zero. 
                if ~isKey( obj.words, currentFeature )
                    featuresCount( i ) = 0;
                    
                else 
                    featuresCount( i ) = obj.words( currentFeature );
                    
                end % if..else
            
            end % for i
            
            % featuresCount = featuresCount / 10;
            
        end % function getFeatureFrequency
        
        % Given newRawData, create a new dataset that uses this new raw
        % data. Make sure to pass 'do not load' as it instruct the
        % constructor not to load data from secondary memory.
        function newDataset = createNewDataset(...
                obj, newRawData, classesCount, newLabels, wordifyFlag )
            
            if nargin <= 4
                wordifyFlag = 0;
            end
            
            newDataset = Database( 'do not load' ); 
            newDataset = setRawData( newDataset, newRawData );     
            newDataset = setNumberOfClasses( newDataset, classesCount );
            newDataset = setLabels( newDataset, newLabels );
            
            newDataset = setParameters(...
                newDataset, obj.likelihoodParameters );
            
            if wordifyFlag == 1
                newDataset = wordify( newDataset );
            end 

        end % function createNewDataset
                        
        % This function extracts all words in the dataset
        % If there are D+1 unique words in the entire dataset
        % Then, our feature vector will have D+1 columns. 
        function obj = wordify( obj )
            
            for i = 1:size( obj.rawData, 1 )
                
                % Column obj.datafield holds the actual text of the 
                % piece of news in row i. I might want to add the
                % title later on.
                currentText = cell2mat( obj.rawData( i, obj.dataField ) );
                
                % Extract all the words in the current text
                % This returns a N by 1 vector, where N is the
                % number of words in the text.
                wordsInNews = textscan( currentText, '%s' );
                
                % Update the frequency of each word accordingly
                obj = updateFrequency( obj, wordsInNews );

            end % for i
            
        end % function wordify
        
        % This given a vector of words, update their frequency 
        % accordingly. First, check if the word is valid. That is
        % it is not too short (less than 3 characters). Then, if
        % this is the first time we encounter the word add it to
        % the dictionary and initialize its frequency to 1. Otherwise,
        % increment the frequency of the word.
        function obj = updateFrequency( obj, words )

            % Process every signle word 
            for i = 1:length( words{:} )

                % A short-cut to the current word.
                % 'currentWord' will be string (not a cell).
                currentWord = words{1}{i};
                
                % The WHILE statement is a mechansim for recursively
                % calling the function removeInvalidCharacters, which
                % resolves multiple issues (e.g., hyphens, question marks,
                % etc.). However, it can resolve only one issue at a time.
                % it must be called repeatedly unitll all issues are
                % resolved.
                while true 
                    
                    % Remove invalid characters such as numbers, emojies,
                    % etc. The first clean word is returned in
                    % 'currentWord.' The remianing of the string is
                    % returned in 'leftOver' which may still contain some
                    % valid words in it and it must be re-examined again.                   
                    [currentWord, leftOver] = ...
                        removeInvalidCharacters( obj, currentWord ); 
                    
                    % If there is not valid word, break.
                    if isempty( currentWord )
                        break
                    end % end if
                    
                    % Skip this word if it is too short. Make an exception
                    % for ? and !'s.
                    if length( currentWord ) <= 2 && ...
                            currentWord( 1 ) ~= '?' &&...
                            currentWord( 1 ) ~= '1'
                        break;
                    end % if
                
                    % Increment the frequency of the current valid word. If
                    % the word is new (i.e., not encountered before) add it
                    % to the dictionary.
                    obj = incrementWordCount( obj, currentWord );
                    
                    % Empty leftOver means there is not need for another
                    % recursive call. Otherwise, the leftOver must be
                    % re-evaluated to see if it contains any valid word.
                    if isempty( leftOver )
                        break; % from while
                        
                    else
                        currentWord = leftOver;
                    end % else...if
                
                end % while

            end % for i                             
            
        end % function eliminateSmallWords
        
        % Remove all impertinent characters from the word
        % e.g., '_', '.', etc. 
        % Also exract '?' and '!' characters as we treat
        % them as words.
        % Also separate hyphenated words
        % The function is not recursive but must be called
        % recursively by the calle as long as there is 
        % something in the 'leftOver' output. 
        function [cleanWord, leftOver] = ...
                removeInvalidCharacters( obj, word )
            
            cleanWord = '';
            indexTracker = 1;
            leftOver = '';
            
            % Treat '?' and '!' as words. They might contain
            % some information. This only cathes the ? and ! 
            % that are in the begining of the word. That 
            % should not be an issue though because the iterative
            % invokation of this function can detect these 
            % characters wherever they are
            if word( 1 ) == '?' || word( 1 ) == '!'
                cleanWord = word( 1 );
                leftOver = word( 2:end );
            end % if
            
            for i = 1:length( word )
                
                % Separate hyphenated words
                % Also exclude ! and ? from the string.                
                % The rest of the string up unitl the hyphen
                % must be parsed separately by recalling this
                % function and passing this leftOver variable 
                % as the new word.                
                if any( word( i ) == ['-', '!', '?', '.'] )
                    leftOver = word( i+1:end );
                    return;
                end % end if      
                                
                
                % Get rid of characters and numbers 
                if ( word( i ) >= 'A' && word( i ) <= 'Z' ) || ...
                        ( word( i ) >= 'a' && word( i ) <= 'z' )
                    cleanWord( indexTracker ) = word( i );
                    indexTracker = indexTracker + 1;
                end % end if
                
            end % for i
            
        end % function removeInvalidCharacters
       
        % Increment the frequency of the current word
        % If it does not exist, add it to the list.
        function obj = incrementWordCount( obj, word )
            
            % Make all words lower case for consistency
            word = lower( word );      
            
            % Do not separate singular and pular words
            % Merely looking at the last two characters is not 
            % the strongest approach but it is simple and good
            % enough (hopefully).
            if word( end ) == 's' && word( end-1 ) == 'e'
                word = word( 1:end-2 );
                
            elseif word( end ) == 's'
                word = word( 1:end-1 );
                
            end % if..else

            % If the word exists add to its frequency
            % Otherwise, add it to the list with default frequency of 0.
            % I use 0 instead of 1 to keep the list sparse.
            if isKey( obj.words, word )
                obj.words( word ) = ...
                    obj.words( word ) + 1;
                               
            else
                obj.words( word ) = 1;
                
            end % if..else
                                    
        end % function incrementWordCount
        
        % Load the databaes
        % The structure of database is a mess
        % So I parse it manually 
        function obj = loadAllData( obj, pathToDataset )
            
            % Read the entire data and save the result 
            % in a big string called 'text.' Keep white
            % spaces
            obj.dataField = 6;
            datasetFile = fopen( pathToDataset );            
            text = fscanf( datasetFile, '%c' );                        
            fclose( datasetFile );
            
            % I know there are 17949 entires in the data.
            % Each entry has 20 columns
            rawDataHolder = cell( 17949, 20 );
            labelsHolder = zeros( 17949, 1  ); 
            
            fieldCounter = 1;
            entryCounter = 1;
            reDoRowFlag = false;
            
            % Go through all characters one by one
            i = 1;
            while i <= length( text )
                
                indexTracker = 1; 
                currentField = '';
                doubleQuoteFlag = false;
                                             
                while true
                              
                    % If end of file is reached, stop
                    if i > length( text ) 
                        break;
                    end
                    
                    % Extract current character
                    currentChar = text( i );
                                               
                    % The last column dose not include a ','
                    % to separate the fields. Instead, we check
                    % for six possible cases or class it has to 
                    % see if we are done reading the string. 
                    if fieldCounter == 20 && ...
                        any( strcmp( currentField,...
                        {'bias', 'fake', 'conspiracy',...
                        'bs', 'satire', 'hate', 'type'} ) )                          
                            i = i + 1;
                            break;
                    end % if
                    
                    % This two consecutive IF statements keep track of
                    % open '"' and close '"'. This is important because
                    % the text field can also contain commas, which can
                    % be confused with field separators. To avoid this,
                    % because text is enclosed between quotation marks,
                    % we keep track of them. It also works interestingly
                    % with nested cases (ish - as long as there is no
                    % comma inside the nested " pair)!!
                    if currentChar == '"' && doubleQuoteFlag == false                            
                        doubleQuoteFlag = true;
                        i = i + 1;
                        continue;
                    end
                    
                    if currentChar == '"' && doubleQuoteFlag == true
                        doubleQuoteFlag = false;
                        i = i + 1;
                        continue;
                    end 
                    
                    % Check if the comma is not inside the text field
                    if currentChar == ',' && doubleQuoteFlag == false
                        i = i + 1;
                        break;                       
                    end        
                    
                    % Discard all non ASCII characters
                    if currentChar <= '!' || currentChar > 'z'
                        currentChar = ' ';
                    end
                    
                    currentField( indexTracker ) = currentChar;
                    indexTracker = indexTracker + 1;
                    i = i + 1;
                    
                end % while    
                                  
                % If this entry does not have any text, it should be
                % discarded. To this end, we re-write the next entry on
                % this one.
                if fieldCounter == obj.dataField 
                    
                        if isempty( strtrim( currentField ) )
                            reDoRowFlag = true;  
                            
                        elseif length( currentField ) <= 100 
                            reDoRowFlag = true;
                        end % if..else
                        
                end % if     
                                               
                % Discard non-english text
                if fieldCounter == 7 && ~strcmp( currentField, 'english' )
                    reDoRowFlag = true;
                end
                
                rawDataHolder( entryCounter, fieldCounter ) = ...
                    {currentField};   
                
                if fieldCounter == 20 
                    
                    labelsHolder( entryCounter ) = ...
                        translateLabel( obj, currentField );
                
                end % if
                
                fieldCounter = fieldCounter + 1;    
                
                if fieldCounter == 21 &&...
                        strcmp( lower( currentField ), 'type' )
                    fieldCounter = 1;
                    continue;
                end
                
                % Reset field counter
                if fieldCounter == 21 && ~reDoRowFlag 
                    fieldCounter = 1;
                    entryCounter = entryCounter + 1;
                end % if
                
                % Reset field counter
                % If the current entry has no text, the next entry should
                % be written in its place. So, we do not increment the
                % entryCounter.
                if fieldCounter == 21 && reDoRowFlag 
                    fieldCounter = 1;
                    reDoRowFlag = false;
                end % if
                
            end % for i
            
            obj.rawData = rawDataHolder( 1:entryCounter-1, : );
            obj.labels = labelsHolder( 1:entryCounter-1, : );
            
        end % function loadAllData
        
        % In binary setting, we consider the bs news target and everything
        % else as non-target
        function label = translateLabel( obj, labelString )
            
            if obj.numberOfClasses == 2
                
                if strcmp( labelString, 'bs' )
                    label = 1;
                    
                else
                    label = 0;
                    
                end % if..else
                
            end % if
            
        end % function translateLabel
        
    end % private utilities
    
end % class Database