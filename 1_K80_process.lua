--F=require 'folder'
require 'functions_processing_training'
require '0_K80_options'

print(opt)


    print("Loading word vectors...")
    glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    raw_data = torch.load(opt.dataPath)
    
	all_data={}
    print("Computing document input representations...")
    all_data.x, all_data.y = preprocess_data(raw_data, glove_table, opt)
	
	--print('==> calling folder')
	--[[folds is going to be a k column matrix, where each column
	--contains the index of validation observations.--]]

	--tr_data.folds=F.KFold((#tr_data.x)[1],opt.valFold):long()

	-- add another shuffle and put .2 in validation




	--medium set for testing
	torch.save(opt.bufferPath,all_data)

	print(all_data)



