require 'kfold'
require 'functions_processing_training'
require '0_K80_options'
require 'model'


print("loading data")
all_tr_data=torch.load(opt.dataPath)
print(all_tr_data)


print('==> shuffling data file')
shuffleIndices = torch.randperm((#all_tr_data.x)[1]):long()
all_tr_data.x=all_tr_data.x:index(1,shuffleIndices)
all_tr_data.y=all_tr_data.y:index(1,shuffleIndices)



--[[
TR={}
VL={}
--for k=1, opt.valFold do

	

	k=1
	print('==> populating fold',k)
	VL.x=all_tr_data.x:index(1,folds[{{},k}])
	VL.y=all_tr_data.y:index(1,folds[{{},k}])

	pointers=torch.ones(#folds)
	pointers:indexFill(2,torch.LongTensor({k}),0)
	tr_pointers=folds:maskedSelect(pointers:byte()):long()

	TR.x=all_tr_data.x:index(1,tr_pointers)
	TR.y=all_tr_data.y:index(1,tr_pointers)

	print('==> calling train_model')
    train_model(model, criterion, TR.x, TR.y, VL.x, VL.y, opt)

	
--]]
