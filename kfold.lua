function folder(n,k)
	local range=torch.range(1,n)
	local giveme=n/k
	local p=torch.ones(n)/n
	folds=torch.zeros(giveme,k)

	for j=1,k do
		local pointers=torch.multinomial(p,giveme,false)
		
		local temp1=range:index(1,pointers)
		folds[{{},j}]=temp1
		local temp2=torch.ones(#range)
		temp2:indexFill(1,pointers,0)
		temp2=temp2:byte()
		range = range:maskedSelect(temp2)
		p=p:maskedSelect(temp2)
	end
	
return folds
end




































