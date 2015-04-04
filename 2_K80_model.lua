require 'torch'
require 'nn'
require 'optim'


   -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.glovePath = "glove/glove.6B.50d.txt" -- path to raw glove data .txt file
	
	
    --opt.dataPath = "data/proc_tr.t7b"
    
	--small
	opt.dataPath = "data/small.t7b"
	
	-- word vector dimensionality
    opt.inputDim = 50 
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.

    -- The Maximal length of a doc, in words
    opt.max_length=30

    opt.nTrainDocs = 130000
    opt.nTestDocs = 0
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 5
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1

    torch.setnumthreads(10)
	
	print("loading data")
    tr_data=torch.load(opt.dataPath)
    print(tr_data)
	
	
    model = nn.Sequential()
	model:add(nn.TemporalConvolution(50, 70, 4, 1))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(3, 1))
	
	model:add(nn.TemporalConvolution(70, 90, 3, 1))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(2, 1))

	model:add(nn.TemporalConvolution(90, 120, 4, 3))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(3, 2))

	model:add(nn.TemporalConvolution(120, 150, 2, 1))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(2, 1))
	
    model:add(nn.Reshape(150, true))	
    model:add(nn.Linear(150, 5))
	model:add(nn.LogSoftMax())
	
    criterion = nn.ClassNLLCriterion()

	print(model)
	
	output=model:forward(tr_data.x)
	print(#output)
	
	
	
	
	