require 'functions_processing_training'

local wordvector_table = torch.load('glove.t7b')
-- this has to change so we read a float model
require 'cunn'
local model = torch.load( 'model_elad_run_simple_epoch_4.net' ):float()
model:evaluate()

local numOfReviews = io.read()
for i = 1, numOfReviews do
    local review = io.read()
    -- opt.max_length is fixed to 100 and word embedding size is fixed to 200.
    local data   = torch.zeros( 100, 200)
    model:evaluate()
    review = preprocess_text(review)
    
    doc_size = 1 
    for word in review:gmatch("%S+") do
    	if doc_size > 100 then  break end
    	if wordvector_table[word] then
    		data[{{ doc_size, {} }}]:add( wordvector_table[word][1] )
            doc_size = doc_size + 1
        else 
        	if wordvector_table[word:gsub("%p+", "")] then
        		data[{{ doc_size, {} }}]:add( wordvector_table[ word:gsub("%p+", "") ][1] )
        		doc_size = doc_size + 1
           end
        end

		local i=1
        while doc_size <= 100 do
			data[{{ doc_size, {} }}]=data[{{ i, {} }}]
			i=i+1
			doc_size=doc_size+1
        end
    end
    print( model:forward(data) )
	--io.write(  )  
end