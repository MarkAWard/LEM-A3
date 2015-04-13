require 'functions_processing_training'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-model', "model_mark_run_shitbags_epoch_3.net")
cmd:option('-dict', "glove.t7b")
cmd:text()
opt = cmd:parse(arg or {})


local wordvector_table = torch.load(opt.dict)
-- this has to change so we read a float model
require 'cunn'
local model = torch.load(opt.model):double()
model:evaluate()

local numOfReviews = io.read()
for i = 1, numOfReviews do
    local review = io.read()
    -- opt.max_length is fixed to 100 and word embedding size is fixed to 200.
    local data   = torch.zeros( 1, 100, 200)
    model:evaluate()
    review = preprocess_text(review)
    
    doc_size = 1 
    for word in review:gmatch("%S+") do
    	if doc_size > 100 then  break end
    	if wordvector_table[word] then
    		data[1][{{ doc_size, {} }}]:add( wordvector_table[word][1] )
            doc_size = doc_size + 1
        else 
        	if wordvector_table[word:gsub("%p+", "")] then
        		data[1][{{ doc_size, {} }}]:add( wordvector_table[ word:gsub("%p+", "") ][1] )
        		doc_size = doc_size + 1
           end
        end

		local i=1
        while doc_size <= 100 do
			data[1][{{ doc_size, {} }}]=data[1][{{ i, {} }}]
			i=i+1
			doc_size=doc_size+1
        end
    end
    local _, argmax = model:forward(data):max(2)
    print( argmax[1] )
end