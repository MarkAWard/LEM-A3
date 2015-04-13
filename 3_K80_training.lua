require 'functions_processing_training'
M = require 'model'
require 'optim'

print(opt)

if opt.type == 'cuda' then
	require 'cunn'
	if opt.machine == 'k80' then
		cutorch.setDevice(opt.gpudevice)
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
if opt.model:match('lookup') == 'lookup' then
	TR.x, TR.y = reviewToIndices(opt.dataPath, glove_table, opt, 'train')
else
	TR.x=torch.load(opt.bufferPath_x)
	TR.y=torch.load(opt.bufferPath_y)
end
print(TR)

VL={}
print('==> creating validation')
if opt.model:match('lookup') == 'lookup' then
	VL.x, VL.y = reviewToIndices(opt.valPath, glove_table, opt, 'val')
else
	VL.x=torch.load(opt.val_x)
	VL.y=torch.load(opt.val_y)
end
print(VL)


print('==> calling train_model')

if opt.type == 'cuda' then
	model=model:cuda()
	criterion=criterion:cuda()
end	

train_model(model, criterion, TR.x, TR.y, VL.x, VL.y, opt)

