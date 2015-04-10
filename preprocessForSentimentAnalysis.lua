-- Commands executed with textwrangler editor
-- 

local stopWordDict = {
"a",
"about",
"after",
"all",
"am",
"an",
"and",
"any",
"are",
"as",
"at",
"be",
"because",
"been",
"being",
"between",
"both",
"but",
"by",
"could",
"did",
"do",
"does",
"doing",
"down",
"during",
"each",
"few",
"for",
"from",
"further",
"had",
"has",
"have",
"having",
"he",
"he'd",
"he'll",
"he's",
"her",
"here",
"here's",
"hers",
"herself",
"him",
"himself",
"his",
"how",
"how's",
"i",
"i'd",
"i'll",
"i'm",
"i've",
"if",
"in",
"into",
"is",
"it",
"it's",
"its",
"itself",
"me",
"my",
"myself",
"of",
"once",
"only",
"or",
"other",
"ought",
"our",
"ours",
"ourselves",
"over",
"own",
"same",
"she",
"she'd",
"she'll",
"she's",
"so",
"than",
"that",
"that's",
"the",
"their",
"theirs",
"them",
"themselves",
"then",
"there",
"there's",
"these",
"they",
"they'd",
"they'll",
"they're",
"they've",
"this",
"those",
"through",
"to",
"until",
"up",
"was",
"we",
"we'd",
"we'll",
"we're",
"we've",
"were",
"what",
"what's",
"when",
"when's",
"where",
"where's",
"which",
"while",
"who",
"who's",
"whom",
"why",
"why's",
"with",
"won't",
"would",
"you",
"you'd",
"you'll",
"you're",
"you've",
"your",
"yours",
"yourself",
"yourselves"}

input = 'trainSample.csv'
local stringx = require('pl.stringx')
require 'ffi'


--[[  FOR TESTING BIG ALLOCATION ON K80
test = torch.randn(100000000,20)
splitPoint = 70000000
testCropped = test:sub(1, splitPoint)
testCropped:size(1) == splitPoint
test[splitPoint]
testCropped[splitPoint]
--]]


local howManyReviews = 8--650000 -- number of reviews in the csv file
local maxSentencesFromEachDocument = 5 -- if K80 can do it, increase it a lot so we can take account for all the sentences.
local maxSeqLength = 20
vocab_idx = 0
vocab_map = {}
local x = torch.zeros( howManyReviews * maxSentencesFromEachDocument,  maxSeqLength )
 idxSentence = 1

function load_csv(fileName)
	fd = io.open(fileName)	
	for review in fd:lines() do
		sentimentLabel = review:sub(1,1)
		-- The input from the csv will be like 1,"document", so we simply extract the document with the following commands.
		review = review:gsub("^...","") -- removes label, comma, opening quotes
		review = review:gsub(".$","") -- removes ending quotes
		 
		-- 1) We mark the capitalized words (which contain more than two characters), with a special token before them.
		-- This way we disambiguate the tone of the word from it's lower case version.
		review = review:gsub(" [A-Z][A-Z]+ ", " next_is_capital%1")
		-- 2) We turn all the words in lower case, so as to capture word similarities between capital and lower case versions.
		review = review:lower()
		-- 3) We remove all the stop words to focus on the essential words.
		for _, stopWord in pairs(stopWordDict) do
		  review = review:gsub(" "..stopWord.." ", " ")
		end
		-- 4) Emoticons are a very important part of sentiment analysis and thus we need to handle them specifically.
		review = review:gsub("[8:=;]['`-]?[%]%)D]", " SMILE "):
			    			gsub("[8:=;]['`-]?[%[%(]", " SADFACE "):
    						gsub("[8:=;]['`-]?[pP]", " LOLFACE "):
			    			gsub("[8:=;]['`-]?[|\\/]", " NEUTRALFACE "):
    						gsub("[8:=;]['`-]?[S]", " CONFUSEDFACE "):
			    			gsub("%^%^", " EVILEARS ")
			    			
		-- 5) handle special symbols
	    review = review:gsub("\\n", " "):
	    					gsub(",", " "):
			      			gsub("!", " EXCLAMATIONMARK "):
			    			gsub("?", " QUESTIONMARK "):
    						gsub("*", " ASTERISK "):
    						gsub("\\\"\"", " DOUBLEQUOTE ")
    
		-- 6) Many consecutive dots reveal frustration, so we group them together, except the last one which shows punctuation.
		review = review:gsub("%.+%."," frustration_dots .")

		-- delete characters that can't be recognized (i.e. UTF) 
		review = review:gsub("\\u00(%w+)", " ")

		----------------------------------------------------------------------------------------------------------------------
		-- what I am about to do, I should seriously think about it again in the morning..
		-- in order to take account of the vanishing/exploding gradients we split each big document review to its sentences.
		-- This way we increase the training sample size and we decouple phrases and their content from the entire review in order
		-- to better capture the statistical language analogies.
		----------------------------------------------------------------------------------------------------------------------
		
		sentences = stringx.split(review, ".")
		for idx, sentence in pairs(sentences) do
			-- A review might have more sentences than the ones we can store. Simply throw them out.
			if idx > maxSentencesFromEachDocument then 
				break
			end
			
			if #sentence ~= 0 then
				local words = stringx.split(sentence)
				for i =1, #words do
					-- We only use the first "maxSeqLength" of words in a sentence
					if i > maxSeqLength then 
						break 
					end
					-- if the word doesn't exist in the dictionary then add it to it.		
					if vocab_map[ words[i]]  == nil then
						vocab_idx = vocab_idx + 1
						vocab_map[ words[i] ] = vocab_idx
					end
					x[idxSentence][i] = vocab_map[ words[i] ]
				end
				idxSentence = idxSentence + 1
			end
		end
	end
	fd:close()	
	-- remove the empty lines we didn't get to use (they are a lot because we pre-allocated the maximum amount of space for speed)
	-- idxSentence was incremented in the end of the loop so that's why we subtract 1
	x = x:sub(1, idxSentence-1) 
	return x	
end
