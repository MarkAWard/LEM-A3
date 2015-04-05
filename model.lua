require 'torch'
require 'nn'
require 'optim'
	
if opt.model=='elad' then
	
    model = nn.Sequential()
	model:add(nn.TemporalConvolution(50, 70, 4, 1))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(3, 1))
	
	model:add(nn.TemporalConvolution(70, 90, 3, 1))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(2, 1))

	model:add(nn.TemporalConvolution(90, 120, 4, 3))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(3, 2))

	model:add(nn.TemporalConvolution(120, 150, 2, 1))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(2, 1))
	
    model:add(nn.Reshape(150, true))	
    model:add(nn.Linear(150, 5))
	model:add(nn.LogSoftMax())
	
    criterion = nn.ClassNLLCriterion()
	
end
