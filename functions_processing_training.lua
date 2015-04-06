require 'torch'
require 'nn'
require 'optim'

ffi = require('ffi')

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)

    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l") 
    --EZ: what is read? 
    --MW: read one line (until newline character)
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                    --EZ: but how does glove_table[word] make sense?
                    --MW: you are creating a lookup dictionary, word --> word_vector_embedding
                else
                    break 
                    --EZ: why a break? 
                    --MW: I guess if there's an empty line just skip it
                end

            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry) 
                --EZ: where is the word and where is the vector??
                --MW: the word is read first when i==1 and does not change till you go to the next line
                --    which is outside of the for loop and i gets reset to 1. the vector is read one element
                --    at a time and placed into the vector
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
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
                        local embedding=wordvector_table[word:gsub("%p+", "")]
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

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

		
	print('==> train')
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
		data=data:index(1,shuffle)
		labels=labels:index(1,shuffle)

	
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
            print("epoch: ", epoch, " batch: ", batch)
        end

--        local accuracy = test_model(model, data, labels, opt)
        local accuracy = num_wrong / labels:size(1)
		print("epoch ", epoch, " tr error: ", accuracy)
		
        local accuracy = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " val error: ", accuracy)

		collectgarbage()
    end
end

function test_model(model, data, labels, opt)

	if opt.type == 'cuda' then
		
		local temp = data
		local temp_labels = labels
			
		data=torch.zeros(#temp):cuda()
		labels=torch.zeros(#temp_labels):cuda()
			
		data[{}]=temp
		labels[{}]=temp_labels
	end

    model:evaluate()

    local pred = model:forward(data)
    local _, argmax = pred:max(2)
    local wrong = torch.ne(argmax:double(), labels:double()):sum() 
    local err = wrong / labels:size(1)

    --local debugger = require('fb.debugger')
    --debugger.enter()

    return err
end
