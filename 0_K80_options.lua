   -- Configuration parameters
    opt = {}
	
	-- model flag
	opt.model = 'elad'
	opt.type = 'cuda'
	
    -- change these to the appropriate data locations
    opt.glovePath = "glove/glove.6B.50d.txt" -- path to raw glove data .txt file
	opt.results = "results/output.csv" 
    -- word vector dimensionality
    opt.inputDim = 50 
    -- The Maximal length of a doc, in words
    opt.max_length=30

    
	opt.dataPath = "data/train.t7b"	

	opt.bufferPath = "data/medium.t7b"
	
	--full
    --opt.nTrainDocs = 130000	
	--small
	--opt.nTrainDocs = 256
	--medium 
	opt.nTrainDocs = 2000
	
    opt.nTestDocs = 0
    opt.nClasses = 5
	opt.valFold=10
	
	-- SGD parameters - play around with these
    opt.nEpochs = 50
    opt.minibatchSize = 128
	opt.total_number=(opt.nClasses*opt.nTrainDocs)
    opt.nBatches = math.floor((opt.total_number-(opt.total_number)/opt.valFold)/ opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1


    torch.setnumthreads(10)
	torch.manualSeed(123)
	
	
