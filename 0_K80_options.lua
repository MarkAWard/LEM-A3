require 'torch' 

-- Configuration parameters
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
-- model flag
cmd:option('-model',  'elad', "name of the model network")
cmd:option('-runName',  'simple'	, "give the job a name")
cmd:option('-type',  'cuda', "type: doubl | float | cuda ")

cmd:option('-machine', 'hpc', "hpc | k80")
cmd:option('-numcores',10, "how many cores?")
cmd:option('-gpudevice',3, "which gpu device do you want?")
cmd:option('-seed',123, "random see")

-- word vector dimensionality
cmd:option('-glovePath',  "glove/glove.6B.200d.txt", "path to glove embedding text file")
cmd:option('-inputDim',  200, "which length embedding")
-- The Maximal length of a doc, in words
cmd:option('-max_length',100, "number of words to keep from reviews, number tensor rows")


cmd:option('-testResult',  "results/output.csv", "output csv ")
cmd:option('-TrainingFolder',  "training/" , "training folder")


cmd:option('-dataPath',  "data/TR_set.csv"	, "training data path")
--	cmd:option('-bufferPath',  "data/full.t7b", "")
cmd:option('-bufferPath_x',  "data/full_x_100x200.t7b", "path to torch binary training data")
cmd:option('-bufferPath_y',  "data/full_y_100x200.t7b", "path to torch binary training data labels")
cmd:option('-proc_glov',  "data/glove.t7b", "path to torch glove binary table")
cmd:option('-sanity',  'data/sanify.csv', "")
--cmd:option('-bufferPath',  "data/medium.t7b", "")

--full
cmd:option('-nTrainDocs',  520000	, "number of reviews")

-- SGD parameters - play around with these
cmd:option('-nEpochs',  10, "MAX")
cmd:option('-minibatchSize',  512, "")
--cmd:option('-total_number', opt.nClasses*opt.nTrainDocs, "")
cmd:option('-learningRate',  0.1, "")
cmd:option('-learningRateDecay',  0.001, "")
cmd:option('-momentum',  0.1, "")
cmd:option('-mode', "options", "options | process | train")

cmd:text()
opt = cmd:parse(arg or {})

opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
opt.idx = 1


torch.setnumthreads(opt.numcores)
torch.manualSeed(opt.seed)


if opt.mode == 'options' then
	return opt
elseif opt.mode == 'process' then
	dofile('1_K80_process.lua')
elseif opt.mode == 'train' then
	dofile('3_K80_training.lua')
else 
	error("Unknown mode")
end

