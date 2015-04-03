-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Please find submission instructions on the handout
------------------------------------------------------------------------
require 'nn'

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)

   local ndims = ((#input)[1] - self.kW) / self.dW + 1

   self.output = torch.zeros(ndims, (#input)[2])

   for j = 1, (#input)[2] do
      idx = 1
      for i = 1, ndims do
         self.output[{i, j}] = torch.log( torch.sum( torch.exp(input[{{idx, idx + self.kW-1}, j}] * self.beta )) / self.kW ) / self.beta
         idx = idx + self.dW
      end    
   end

   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)

   self.gradInput = torch.zeros(input:size())

   for col = 1, (#gradOutput)[2] do
      for i = 1, (#gradOutput)[1] do
         local denom = torch.sum( torch.exp( input[ {{i, i + self.kW - 1}, col}] * self.beta ) )
         j_start = self.dW * (i - 1) + 1
         for j = j_start, j_start + self.kW-1 do
            num = torch.exp(  input[{j, col}] * self.beta )
            self.gradInput[{j, col}] = self.gradInput[{j, col}] + ( gradOutput[{i, col}] * (num / denom) )
         end
      end
   end

   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end

mod = nn.TemporalLogExpPooling(3,1,2)

input_10_3 = torch.randn(10,3)
input_4_5 = torch.randn(4,5)
out_10_3 = mod:forward(input_10_3)
out_4_5 = mod:forward(input_4_5)

df1 = torch.randn(8,1)
df2 = torch.randn(8,3)
input = torch.randn(10,1)
di1 = mod:backward(input, df1)
di2 = mod:backward(input_10_3, df2)


mod2 = nn.TemporalLogExpPooling(3,2,2)

input_7_3 = torch.randn(7,3)
input_9_5 = torch.randn(9,5)
out_7_3 = mod2:forward(input_7_3)
out_9_5 = mod2:forward(input_9_5)

df11 = torch.randn(3,3)
df22 = torch.randn(4,5)
di11 = mod2:backward(input_7_3, df11)
di22 = mod2:backward(input_9_5, df22)
