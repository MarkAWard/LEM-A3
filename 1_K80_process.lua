--F=require 'folder'
require 'functions_processing_training'
require '0_K80_options'

print(opt)


    print("Loading word vectors...")
    glove_table, dictionay_size = load_glove(opt.glovePath, opt.inputDim)
    
    
    print("Computing document input representations...")
    all_data.x, all_data.y = load_train_csv(opt.dataPath, glove_table, opt)

	--medium set for testing
	torch.save(opt.bufferPath,all_data)

	print(all_data)



