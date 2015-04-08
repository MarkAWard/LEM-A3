   -- Configuration parameters
    opt = {}
	
	-- model flag
	opt.model = 'smallerboy_200'
	opt.runName = 'trying'	
	opt.type = 'cuda'
	
    -- change these to the appropriate data locations
--    opt.glovePath = "glove/glove.6B.50d.txt"
    opt.glovePath = "glove/glove.6B.200d.txt"

    -- word vector dimensionality
    opt.inputDim = 200
    -- The Maximal length of a doc, in words
    opt.max_length=30



	opt.testResult = "results/output.csv"
	opt.TrainingFolder = "training/" 

    
	opt.dataPath = "data/train.t7b"	
--	opt.bufferPath = "data/full.t7b"
	opt.bufferPath = "data/full_200.t7b"
	--opt.bufferPath = "data/medium.t7b"
	
	--full
    opt.nTrainDocs = 130000	
	--small
	--opt.nTrainDocs = 256
	--medium 
	--opt.nTrainDocs = 10000
	
    opt.nTestDocs = 0
    opt.nClasses = 5
	opt.valFold=10
	
	-- SGD parameters - play around with these
    opt.nEpochs = 50
    opt.minibatchSize = 512
	opt.total_number=(opt.nClasses*opt.nTrainDocs)
    opt.nBatches = math.floor((opt.total_number-(opt.total_number)/opt.valFold)/ opt.minibatchSize)
    opt.learningRate = 0.001
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1

	opt.numcores=10
	opt.gpudevice=4
	
	opt.seed=123

	
    torch.setnumthreads(opt.numcores)
	torch.manualSeed(opt.seed)
	
	
