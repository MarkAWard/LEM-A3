require 'functions_processing_training'
M = require 'model'
require 'optim'

print(opt)

if opt.type == 'cuda' then
	require 'cunn'
	if opt.machine == 'k80' then
		cutorch.setDevice(opt.device)
	else
		cutorch.setDevice(1)
	end
	cutorch.getDeviceProperties(cutorch.getDevice())
end 


if opt.model:match('lookup') == 'lookup' then
	glove_table, dictionay_size = load_glove(opt.glovePath, opt.inputDim)
end

model, criterion = M:select_model(opt, dictionay_size)

init_model(model, glove_table, opt)


TR={}
print("loading data")
TR.x=torch.load(opt.bufferPath_x)
TR.y=torch.load(opt.bufferPath_y)
print(TR)

VL={}
print('==> creating validation')
VL.x=torch.load(opt.val_x)
VL.y=torch.load(opt.val_y)
print(VL)


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



