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
print(all_tr_data)


print('==> creating validation')

TR={}
VL={}


shuffle = torch.randperm((#all_tr_data.x)[1]):long()

for i=1, shuffle:size(1) do
	all_tr_data.x[{i,{},{}}]=all_tr_data.x[{shuffle[i],{},{}}]
	all_tr_data.y[i]=all_tr_data.y[shuffle[i]]
end

temp=.9*(all_tr_data.x:size(1))
VL.x=all_tr_data.x[{{temp+1,(all_tr_data.x:size(1))},{},{}}]
VL.y=all_tr_data.y[{{temp+1,(all_tr_data.x:size(1))}}]


TR.x=all_tr_data.x[{{1,temp},{},{}}]
TR.y=all_tr_data.y[{{1,temp}}]


print('==> calling train_model')

if opt.type == 'cuda' then
	model=model:cuda()
	criterion=criterion:cuda()
end	

--print('==> loading previous model')
--model=torch.load('training/model_smallerboy_200_run_trying_fold_1_epoch_4.net' )

train_model(model, criterion, TR.x, TR.y, VL.x, VL.y, opt)




	--test_model(model, VL.x, VL.y, opt)
--]]



