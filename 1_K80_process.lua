F=require 'kfold'
require 'functions_processing_training'
require '0_K80_options'

	print(opt)

    print("Loading word vectors...")
    glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    raw_data = torch.load(opt.dataPath)
    
	tr_data={}
    print("Computing document input representations...")
    tr_data.x, tr_data.y = preprocess_data(raw_data, glove_table, opt)
	
	print('==> calling folder')
	--[[folds is going to be a k column matrix, where each column
	contains the index of validation observations.--]]
	tr_data.folds=F.KFold((#tr_data.x)[1],opt.valFold)


	--medium set for testing
	torch.save(opt.bufferPath,tr_data)

	print(tr_data)
