require 'torch'
ffi = require 'ffi'
require 'functions_processing_training.lua'

train = torch.load('data/train.t7b')
glove = load_glove('glove/glove.6B.200d.txt', 200)

f = io.open('missed44.txt', 'w')

for i=1, 5 do 
	for j=1, 130000 do
		local index = train.index[i][j]
		local document = ffi.string(torch.data(train.content:narrow(1, index, 1))):lower()
		document = document:gsub("\\n", " "):gsub("n't ", " not "):gsub("'re ", " are "):gsub("'ve ", " have "):gsub("'ll ", " will "):gsub("'d ", " would "):gsub("'s ", " ")
		document = document:gsub("[8:=;]['`-]?[%]%)D]", " SMILE "):gsub("[8:=;]['`-]?[%[%(]", " SADFACE "):gsub("[8:=;]['`-]?[pP]", " LOLFACE "):gsub("[8:=;]['`-]?[|\\/]", " NEUTRALFACE "):gsub("[-+]?[.%d]*[%d]+[:,.%d]*", "NUMBER")

		for word in document:gmatch("%S+") do
			if not glove[word:gsub("%p+", "")] then
				f:write(word:gsub("%p+", "") .. "\t" .. word .. "\n")
			end
		end

	end
end

f:close()

