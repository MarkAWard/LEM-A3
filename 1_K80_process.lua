--F=require 'folder'
require 'functions_processing_training'
require '0_K80_options'

print(opt)


    print("Loading word vectors...")
    glove_table, dictionay_size = load_glove(opt.glovePath, opt.inputDim)
    

	print("Allocating Memory...")
	all_data={}
	--all_data.x=torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.max_length, opt.inputDim)
	--all_data.y=torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    print("Computing document input representations...")
    all_data.x, all_data.y = load_train_csv(opt.dataPath, glove_table, opt)

	
	torch.save(opt.bufferPath_x,all_data.x)
	torch.save(opt.bufferPath_y,all_data.y)

	torch.save(opt.proc_glov,glove_table)


	print(all_data)



