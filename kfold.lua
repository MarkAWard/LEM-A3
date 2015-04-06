local CrossVal = {}

function CrossVal.KFold(n,k)
	folds = {}
	local range = torch.randperm(n):long()
	local giveme = n / k

	for i = 1, k do
		folds[i] = range[ {{ (i-1) * giveme + 1, i * giveme }} ]
	end

	return folds
end

return CrossVal