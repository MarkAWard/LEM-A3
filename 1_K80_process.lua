require 'kfold'
require 'functions_processing_training'
require '0_K80_options'

    print("Loading word vectors...")
    glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    processed_data, labels = preprocess_data(raw_data, glove_table, opt)

    tr_data={}
	tr_data.x=processed_data
	tr_data.y=labels
	
	
	print('==> calling folder')
	--[[folds is goind to be a k column matrix, where each column
	contains the index of validation observations.--]]
	tr_data.folds=folder((#tr_data.x)[1],opt.valFold):long()
	torch.save('data/folds.t7b',folds)
	
	
	--torch.save('data/proc_tr.t7b',tr_data)
    
	--small set for testing
	torch.save('data/small.t7b',tr_data)

