require 'functions_processing_training'
require '0_K80_options'
require 'model'


print("loading data")
all_tr_data=torch.load(opt.bufferPath)
print(all_tr_data)

--[[
THIS SHOULD BE IN THE TRAINING FUNCTION.
IT DOES NOT MAKE SENSE TO DO IT HERE JUST ONCE
print('==> shuffling data file')
shuffleIndices = torch.randperm((#all_tr_data.x)[1]):long()
all_tr_data.x=all_tr_data.x:index(1,shuffleIndices)
all_tr_data.y=all_tr_data.y:index(1,shuffleIndices)
--]]


TR={}
VL={}
--for k=1, opt.valFold do

	

	k=1
	print('==> populating fold',k)
	VL.x=all_tr_data.x:index(1,all_tr_data.folds[{{},k}])
	VL.y=all_tr_data.y:index(1,all_tr_data.folds[{{},k}])

	pointers=torch.ones(#all_tr_data.folds)
	pointers:indexFill(2,torch.LongTensor({k}),0)
	
	tr_pointers=all_tr_data.folds:maskedSelect(pointers:byte()):long()

	TR.x=all_tr_data.x:index(1,tr_pointers)
	TR.y=all_tr_data.y:index(1,tr_pointers)

	print('==> calling train_model')
	
    train_model(model, criterion, TR.x, TR.y, VL.x, VL.y, opt)

--	F.train(model, criterion, TR.x, TR.y, VL.x, VL.y, opt)
--]]



