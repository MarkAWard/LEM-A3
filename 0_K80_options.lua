   -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.glovePath = "glove/glove.6B.50d.txt" -- path to raw glove data .txt file
    
	
	--opt.dataPath = "data/train.t7b"
	opt.dataPath = "data/proc_tr.t7b"
	--opt.dataPath = "data/small.t7b"
	
    -- word vector dimensionality
    opt.inputDim = 50 
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.

    -- The Maximal length of a doc, in words
    opt.max_length=30

    --opt.nTrainDocs = 130000
	
	--small
	opt.nTrainDocs = 256
	
    opt.nTestDocs = 0
    opt.nClasses = 5
    
	opt.valFold=10
	
	-- SGD parameters - play around with these
    opt.nEpochs = 50
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1


    torch.setnumthreads(10)
