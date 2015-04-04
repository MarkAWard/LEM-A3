----------------------------------------------------------------------------
--  Gradient Checker 
--  Check that all gradients of a model are computed correctly by using
--  a numerical approximation of each gradient and comparing it to the 
--  gradients that the model finds
----------------------------------------------------------------------------

require 'nn'

function compute_gradient(model, original, crit, input, labels, idx, eps) 

	params, _ = model:getParameters()

	params[idx] = original[idx] + eps
	model_out1 = model:forward(input)
	print(idx, params[idx])
	params[idx] = original[idx] - eps
	model_out2 = model:forward(input)
	print(idx, params[idx])

	params = original

	E1 = crit:forward(model_out1, labels)
	E2 = crit:forward(model_out2, labels)

	print(idx, E1, E2)

	return (E1 - E2) / (2 * eps)

end

model = nn.Sequential()
model:add(nn.TemporalConvolution(1, 20, 10, 1))
---------
model:add(nn.TemporalMaxPooling(3, 1))
---------
model:add(nn.Reshape(20*39, true))
model:add(nn.Linear(20*39, 5))
model:add(nn.LogSoftMax())

crit = nn.ClassNLLCriterion()

model:zeroGradParameters()
parameters, gradparams = model:getParameters()
original = parameters:clone()

input = torch.randn(5,50,1)
labels = torch.Tensor{1,2,2,5,5}

output = model:forward(input)
grads = crit:backward(output, labels)
model:backward(input, grads)

eps = 0.1
--for i = 1, (#original)[1] do
for i = 1, 20 do

	approx_grad = compute_gradient(model, original, crit, input, labels, i, eps)
	print(gradparams[i], approx_grad)

end
