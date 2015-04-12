require 'torch'
require 'nn'
require 'optim'
local stringx = require('pl.stringx')
ffi = require('ffi')
stopWords = require('stopwords.lua')

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim, limited)

    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l") 
    --EZ: what is read? 
    --MW: read one line (until newline character)
    local counter = 1
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:lower() -- change to lower case
                if string.len(word) > 0 then

                    -- cut down the size of glove that needs to be loaded/saved
                    -- only keep words that we would actually find after text preprocessing
                    if limited then 
                        -- process the text, strip whitesapce, and check if its the same word
                        if preprocess_text(word):match("%S+") == word then
                            glove_table[word] = {torch.zeros(inputDim, 1), counter} -- padded with an extra dimension for convolution
                            counter = counter + 1
                        else -- the word was different so we would never find it in the processed text, skip it
                            break
                        end
                    else -- ORIGINAL load everything
                        glove_table[word] = {torch.zeros(inputDim, 1), counter} -- padded with an extra dimension for convolution
                        counter = counter + 1
                    end
                    --EZ: but how does glove_table[word] make sense?
                    --MW: you are creating a lookup dictionary, word --> word_vector_embedding
                else
                    break 
                    --EZ: why a break? 
                    --MW: I guess if there's an empty line just skip it
                end

            else
                -- read off and store each word vector element
                glove_table[word][1][i-1] = tonumber(entry) 
                --EZ: where is the word and where is the vector??
                --MW: the word is read first when i==1 and does not change till you go to the next line
                --    which is outside of the for loop and i gets reset to 1. the vector is read one element
                --    at a time and placed into the vector
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table, counter-1
end

--- Here we simply encode each document as a fixed-length vector 
-- by computing the unweighted average of its word vectors.
-- A slightly better approach would be to weight each word by its tf-idf value
-- before computing the bag-of-words average; this limits the effects of words like "the".
-- Still better would be to concatenate the word vectors into a variable-length
-- 2D tensor and train a more powerful convolutional or recurrent model on this directly.


function preprocess_data(raw_data, wordvector_table, opt)

   --wordvector_table=glove_table

    -- opt.max_length is going to be the maximum length of a document.
    -- OPEN ISSUE: we can either zero pad if the document is too short,
    -- or do like Collbert et al and plug a repeating code word "END".
    -- THE CURRENT implementation is zeros.

    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.max_length,opt.inputDim)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))

    --i=1
    --j=1
    
    
    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            -- break each review into words and compute the document average
            for word in document:gmatch("%S+") do
                if wordvector_table[word:gsub("%p+", "")] then
                    if (doc_size<opt.max_length) then 
                        --print(#document)
                        local embedding=wordvector_table[word:gsub("%p+", "")][1]
                        data[k][{{doc_size,{}}}]:add(embedding)
                        doc_size = doc_size+1
                    end
                end
            end
            labels[k] = i
        end
    end
    --data=data:reshape(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.max_length)
    return data, labels
end


function load_train_csv( filename, wordvector_table, opt)
    
    local data   = torch.zeros(opt.nTrainDocs, opt.max_length, opt.inputDim)
    local labels = torch.zeros(opt.nTrainDocs)

    k = 1
    f = io.open(filename, 'r')
    for line in f:lines() do
        -- The input from the csv will be like 1,"document", so we simply extract the document with the following commands.
        labels[k] = line:sub(1,1)
        local review = line:gsub("^..."," "):gsub(".$"," ") -- removes label, comma, and leading/trailing quotes
        -- use the preprocessing defined below
        review = preprocess_text(review)

        doc_size = 1 
		if k % 1000 ==0 then print(k) end

        for word in review:gmatch("%S+") do
			if (doc_size > opt.max_length) then 
				break
			end
            if wordvector_table[word] then
                data[k][{{ doc_size, {} }}]:add( wordvector_table[word][1] )
                doc_size = doc_size + 1
            else 
                if wordvector_table[word:gsub("%p+", "")] then
                    data[k][{{ doc_size, {} }}]:add( wordvector_table[ word:gsub("%p+", "") ][1] )
                    doc_size = doc_size + 1
            -- else
            --     pass
                end
            end
        end

        -- if the length of the review was less than the max length allowed then
        -- the repeat the review text in the same order till we hit the max length
		i=1
        while doc_size <= opt.max_length do
			data[k][{{ doc_size, {} }}]=data[k][{{ i, {} }}]
			i=i+1
			doc_size=doc_size+1
            --data[k][{{ doc_size, {} }}]:add( data[k][{{ (doc_size - 1) % len + 1, {} }}] )
        end
        k = k + 1
    end
    f:close()

    return data, labels
end



function preprocess_text(text)

    text = text:gsub("^\"", " "):gsub("\"$", " "):
                gsub("\\n", " "):gsub("\\\"\"", " ")

    text = text:gsub("http%S+", " url "):
                gsub("www%S+", " url ")
    
    text = text:gsub("(%a+)(%p+) ", "%1 %2 "):
                gsub(" (%p+)(%a+) ", " %1 %2 ")

    text = text:gsub(" [A-Z][A-Z]+ ", " intense %1")

    text = text:lower()

    text = text:gsub("can't", " not "):
                gsub("won't", " not "):
                gsub("n't ", " not "):
                gsub("'re ", " "):
                gsub("'ve ", " "):
                gsub("'ll ", " "):
                gsub("'d ", " "):
                gsub("'s ", " ")

    text = text:gsub("[8:=;]['`-]?[%]%)d]", " :) "):
                gsub("[8:=;]['`-]?[%[%(]", " :( "):
                gsub("[8:=;]['`-]?[p]", " :p "):
                gsub("[8:=;]['`-]?[|\\/]", " :| ") 

    text = text:gsub("$[.]?[%d]+[,.%d]*", " money ") 

    -- repeated punctuation 
    punct = {"!", "%?", "!%?", "%.", "%-"}
    for _, p in pairs(punct) do
        text = text:gsub("["..p.."]".."["..p.."]".."["..p.."]+", " repeat " .. p .. " ")
    end

    -- elongated words only do a subset of letter
    letters = {"m", "y", "w", "g", "h"}
    for _, l in pairs(letters) do
        prev = string.char(l:byte()-1)
        nxt = string.char(l:byte()+1)
        text = text:gsub(" ([a-"..prev..nxt.."-z]+)"..l..l..l.."+ ", " very %1"..l.." ")
    end
    text = text:gsub(" soo+ ", " very so "):
                gsub(" noo+ ", " very no ")

    for _, stopWord in pairs(stopWords) do
        text = text:gsub(" "..stopWord.." ", " ")
    end

    return text
end


function train_model(model, criterion, data, labels, test_data, test_labels, opt)

		
	print('==> train')
	metric_holder=torch.zeros(opt.nEpochs,2)
    parameters, grad_parameters = model:getParameters()

    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        model:training()
		
        
		
		if opt.type == 'cuda' then
			
			local temp_minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
			local temp_minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
			
			
			minibatch=torch.zeros(#temp_minibatch):cuda()
			minibatch_labels=torch.zeros(#temp_minibatch_labels):cuda()
			
			
			minibatch[{}]=temp_minibatch
			minibatch_labels[{}]=temp_minibatch_labels
			
		else 
			local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
			local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
		end
		
		
        local pred = model:forward(minibatch)
		
        local minibatch_loss = criterion:forward(model.output, minibatch_labels)
		
		
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        local _, argmax = pred:max(2)
        num_wrong = num_wrong + torch.ne(argmax:double(), minibatch_labels:double()):sum()

        return minibatch_loss, grad_parameters
    end
    
    for epoch=1, opt.nEpochs do

        num_wrong = 0    
	
		-- Shuffling the training data   
		shuffle = torch.randperm((#data)[1]):long()

		for i=1, shuffle:size(1) do
			data[{i,{},{}}]=data[{shuffle[i],{},{}}]
			labels[i]=labels[shuffle[i]]
		end

		--data=data:index(1,shuffle)
		--labels=labels:index(1,shuffle)

	
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
			if batch % 20 ==0 then 
				print("epoch: ", epoch, " batch: ", batch) 
				collectgarbage()
			end
        end

		collectgarbage()

		local filename = paths.concat(opt.TrainingFolder,'model_' .. opt.model .. '_run_' .. opt.runName .. '_epoch_' .. epoch .. '.net')
		torch.save(filename, model)

        local accuracy_tr = num_wrong / labels:size(1)
		print("epoch ", epoch, " tr error: ", accuracy_tr)
		
        local accuracy_vl = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " val error: ", accuracy_vl)


		metric_holder[epoch][1]=accuracy_tr
		metric_holder[epoch][2]=accuracy_vl

		collectgarbage()
    end
	local filename = paths.concat(opt.TrainingFolder,'ModelMetrics_' .. opt.model .. '_run_' .. opt.runName .. '.t7b')
	torch.save(filename,metric_holder)
end

function test_model(model, data, labels, opt)


	model:evaluate()
	local n = (data:size())[1]
	local no_wrong = 0
	
	for t = 1, n, opt.minibatchSize do
		
		if opt.type == 'cuda' then
			local temp  = data[ {{ t, math.min(t+opt.minibatchSize-1, n) }} ]
			local temp_targets = labels[ {{ t, math.min(t+opt.minibatchSize-1, n) }} ]		
		
			inputs=torch.zeros(#temp):cuda()
			targets=torch.zeros(#temp_targets):cuda()			
			
			inputs[{}]=temp
			targets[{}]=temp_targets
    	else
			inputs  = data[ {{ t, math.min(t+opt.minibatchSize-1, n) }} ]
			targets = labels[ {{ t, math.min(t+opt.minibatchSize-1, n) }} ]		
		end
		
    	local output = model:forward(inputs)
    	local trash, argmax = output:max(2)
    	no_wrong = no_wrong + torch.ne(argmax, targets):sum()
    end

	-- return error
    return no_wrong/n
end


--[[
-- Model: The model to be processed. Assumes that the model has a LookupTable layer as its first layer.
-- Dict: The dictionary which contains the word embeddings and their respective indexes.
-- EmbeddingSize: The size of a word's embedding.
--]]
function init_model(model, dict, opt)
	if opt.model == 'lookup_elad' then
		for key,val in pairs(dict) do                                                               
			model:get(1):getParameters()[ {{ (val[2]-1) * opt.inputDim + 1, val[2] * opt.inputDim }} ] = val[1]
		end
	end
end


function reviewToIndices(inputFile, glove, opt)
    
    local data   = torch.zeros(opt.nTrainDocs, opt.max_length)
    local labels = torch.zeros(opt.nTrainDocs)
	
	local fd = io.open(inputFile)	

	k=1
	for review in fd:lines() do
		labels[k] = review:sub(1,1)
		review = review:gsub("^..","") -- removes label, comma
		words = stringx.split(review)
		doc_size = 1
		
		for _, word in pairs(words) do
			if doc_size > opt.max_length then break end
			-- drop if not in glove 
			if glove[word] then 
				data[k][doc_size] = glove[word][2]
				doc_size = doc_size + 1
            else 
                if glove[word:gsub("%p+", "")] then
                    data[k][doc_size] = glove[ word:gsub("%p+", "") ][2] 
                    doc_size = doc_size + 1
                end
            end
        end
        
        i=1
        while doc_size <= opt.max_length do
			data[k][doc_size]=data[k][i]
			i=i+1
			doc_size=doc_size+1
        end
        k = k + 1    
	end
	return data, labels
end
