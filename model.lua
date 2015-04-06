	
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


if opt.model=='bigboy' then
	
	--bigrams
    bigrams = nn.Sequential()
	bigrams:add(nn.TemporalConvolution(50, 70, 2, 1))
	bigrams:add(nn.ReLU())
	bigrams:add(nn.TemporalConvolution(70, 90, 2, 1))
	bigrams:add(nn.ReLU())
	bigrams:add(nn.TemporalMaxPooling(3, 1))
	bigrams:add(nn.TemporalConvolution(90, 110, 2, 1))
	bigrams:add(nn.ReLU())
	bigrams:add(nn.TemporalMaxPooling(4, 1))

	
--	output=bigrams:forward(all_tr_data.x)
--	print(#output)
	--10000
    --22
    --110
	
	
	--trigrams
    trigrams = nn.Sequential()
	trigrams:add(nn.TemporalConvolution(50, 70, 3, 1))
	trigrams:add(nn.ReLU())
	trigrams:add(nn.TemporalConvolution(70, 90, 3, 1))
	trigrams:add(nn.ReLU())
	trigrams:add(nn.TemporalMaxPooling(3, 1))
	trigrams:add(nn.TemporalConvolution(90, 110, 2, 1))
	trigrams:add(nn.ReLU())
	trigrams:add(nn.TemporalMaxPooling(2, 1))
	
--	output=trigrams:forward(all_tr_data.x)
--	print(#output)
	--10000
    --22
    --110

	
	
	--quadgrams
    quadgrams = nn.Sequential()
	quadgrams:add(nn.TemporalConvolution(50, 80, 4, 1))
	quadgrams:add(nn.ReLU())
	quadgrams:add(nn.TemporalConvolution(80, 110, 4, 1))
	quadgrams:add(nn.ReLU())
	quadgrams:add(nn.TemporalMaxPooling(3, 1))
	
--	output=quadgrams:forward(all_tr_data.x)
--	print(#output)
	--10000
    --22
    --110
	
	
	par=nn.Concat(2)
	par:add(bigrams)
	par:add(trigrams)
	par:add(quadgrams)

    model = nn.Sequential()
	model:add(par)
	
	model:add(nn.TemporalConvolution(110, 150, 4, 2))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(3, 2))
	
	model:add(nn.TemporalConvolution(150, 170, 4, 3))
	model:add(nn.ReLU())
	model:add(nn.TemporalMaxPooling(3, 1))

	model:add(nn.TemporalConvolution(170, 200, 2, 1))
	model:add(nn.ReLU())
	
	model:add(nn.Reshape(200, true))	
    model:add(nn.Linear(200, 5))
	model:add(nn.LogSoftMax())
	
    criterion = nn.ClassNLLCriterion()
end

