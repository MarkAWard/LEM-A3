   -- Configuration parameters
    opt = {}
	
	-- model flag
	opt.model = 'lookup_elad'
	opt.runName = 'simple'	
	opt.type = 'cuda'

	opt.numcores=10
	opt.gpudevice=3
	opt.seed=123
	
    opt.glovePath = "glove/glove.6B.200d.txt"

    -- word vector dimensionality
    opt.inputDim = 200
    -- The Maximal length of a doc, in words
    opt.max_length=100

	opt.testResult = "results/output.csv"
	opt.TrainingFolder = "training/" 

    
--	opt.dataPath = "data/TR_set.csv"	
	opt.dataPath = "data/test.csv"	

	opt.bufferPath_x = "data/test_x_100x200.t7b"
	opt.bufferPath_y = "data/test_y_100x200.t7b"




--	opt.bufferPath_x = "data/full_x_100x200.t7b"
--	opt.bufferPath_y = "data/full_y_100x200.t7b"
	opt.proc_glov = "data/glove.t7b"
	opt.sanity = 'data/sanify.csv'
	--opt.bufferPath = "data/medium.t7b"
	
	--full
    opt.nTrainDocs = 104000	
	opt.nTestDocs = 0
    --opt.nTestDocs = 0
    opt.nClasses = 5
	opt.valFold=10
	
	-- SGD parameters - play around with these
    opt.nEpochs = 10
    opt.minibatchSize = 512
	opt.total_number=(opt.nClasses*opt.nTrainDocs)
    opt.nBatches = math.floor((opt.total_number-(opt.total_number)/opt.valFold)/ opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1


	
    torch.setnumthreads(opt.numcores)
	torch.manualSeed(opt.seed)
	
	
