require 'functions_processing_training'
require '0_K80_options'

fin=io.open(opt.dataPath,'r')
fout=io.open(opt.sanity,'w')

i=1
for line in fin:lines() do

    local label = line:sub(1,1)
    local review = line:gsub("^..."," "):gsub(".$"," ")
	review = preprocess_text(review)
	fout:write(label .. ',' .. review .. '\n')

	if i % 1000 == 0 then print(i) end
	i=i+1
end


fin:close()
fout:close()


