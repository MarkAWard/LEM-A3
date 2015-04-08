require 'functions_processing_training'
require '0_K80_options'
require 'model'
require 'optim'

print(opt)

if opt.type == 'cuda' then
	require 'cunn';
	cutorch.setDevice(opt.gpudevice)
	cutorch.getDeviceProperties(cutorch.getDevice())
end


print("loading data")
all_tr_data=torch.load(opt.bufferPath)
all_tr_data.folds=torch.load('data/folds.t7b')
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


	TR.x=torch.zeros(585000,30,200)

-- this crushes memory
--	TR.x=all_tr_data.x:index(1,tr_pointers)

-- This should work. doing a loop instead.
--	TR.x:indexCopy(1,tr_pointers,all_tr_data.x)
	

	for i=1, tr_pointers:size(1) do
		TR.x[{i,{},{}}]=all_tr_data.x[{tr_pointers[i],{},{}}]
	end

	TR.y=all_tr_data.y:index(1,tr_pointers)

	print('==> calling train_model')

	if opt.type == 'cuda' then
		model=model:cuda()
		criterion=criterion:cuda()
	end	

	--model=torch.load('training/model_smallerboy_run_testing_fold_1_epoch_7.net' )

	train_model(model, criterion, TR.x, TR.y, VL.x, VL.y, opt)




	--test_model(model, VL.x, VL.y, opt)
--]]



