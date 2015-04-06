require 'functions_processing_training'
require '0_K80_options'
require 'model'
require 'optim'

print(opt)

if opt.type == 'cuda' then
	require 'cunn';
	cutorch.setDevice(3)
	cutorch.getDeviceProperties(cutorch.getDevice())
end


print("loading data")
all_tr_data=torch.load(opt.bufferPath)
print(all_tr_data)


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

	if opt.type == 'cuda' then
		model=model:cuda()
		criterion=criterion:cuda()
	end	

	
    train_model(model, criterion, TR.x, TR.y, VL.x, VL.y, opt)

--]]



