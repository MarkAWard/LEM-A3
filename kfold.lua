function folder(n,k)
	local range=torch.range(1,n)
	local giveme=n/k
	local p=torch.ones(n)/n
	local folds=torch.zeros(giveme,k)

	for j=1,k do
		print('___creating fold',j)
		print('___multinomial')
		local pointers=torch.multinomial(p,giveme,false)
		
		print('___indexing')
		local temp1=range:index(1,pointers)
		folds[{{},j}]=temp1
		local temp2=torch.ones(#range)
		
		print('___filling')
		temp2:indexFill(1,pointers,0)
		temp2=temp2:byte()
		
		print('___masked select')
		range = range:maskedSelect(temp2)
		p=p:maskedSelect(temp2)
	end
	
return folds
end




































