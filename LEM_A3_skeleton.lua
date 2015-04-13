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

   local mode = #(#input)
   if mode == 2 then
      batch_size = 1
      ndims = ((#input)[1] - self.kW) / self.dW + 1
      cols = (#input)[2]
      self.output = torch.zeros(ndims, cols)
   elseif mode == 3 then
      batch_size = (#input)[1]
      ndims = ((#input)[2] - self.kW) / self.dW + 1
      cols = (#input)[3]
      self.output = torch.zeros(batch_size, ndims, cols)
   end

   local idx = 1
   for k = 1, batch_size do
      for j = 1, cols do
         idx = 1
         for i = 1, ndims do
            if mode == 2 then
               self.output[{i, j}] = torch.log( torch.sum( torch.exp(input[{{idx, idx + self.kW-1}, j}] * self.beta )) / self.kW ) / self.beta
            else
               self.output[k][{i, j}] = torch.log( torch.sum( torch.exp(input[k][{{idx, idx + self.kW-1}, j}] * self.beta )) / self.kW ) / self.beta
            end
            idx = idx + self.dW
         end    
      end
   end

   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)

   local mode = #(#gradOutput)
   if mode == 2 then
      batch_size = 1
      ndims = (#gradOutput)[1]
      cols = (#gradOutput)[2]
   elseif mode == 3 then
      batch_size = (#gradOutput)[1]
      ndims = (#gradOutput)[2]
      cols = (#gradOutput)[3]
   end
   self.gradInput = torch.zeros(input:size())

   local denom = 1
   for k = 1, batch_size do
      for col = 1, cols do
         for i = 1, ndims do
            if mode == 2 then
               denom = torch.sum( torch.exp( input[ {{i, i + self.kW - 1}, col}] * self.beta ) )
            else
               denom = torch.sum( torch.exp( input[k][ {{i, i + self.kW - 1}, col}] * self.beta ) )
            end
            j_start = self.dW * (i - 1) + 1
            for j = j_start, j_start + self.kW-1 do
               if mode == 2 then
                  num = torch.exp(  input[{j, col}] * self.beta )
                  self.gradInput[{j, col}] = self.gradInput[{j, col}] + ( gradOutput[{i, col}] * (num / denom) )
               else
                  num = torch.exp(  input[k][{j, col}] * self.beta )
                  self.gradInput[k][{j, col}] = self.gradInput[k][{j, col}] + ( gradOutput[k][{i, col}] * (num / denom) )
               end
            end
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

-- mod = nn.TemporalLogExpPooling(3,1,2)

-- input_10_3 = torch.randn(10,3)
-- input_4_5 = torch.randn(4,5)
-- out_10_3 = mod:forward(input_10_3)
-- out_4_5 = mod:forward(input_4_5)

-- df1 = torch.randn(8,1)
-- df2 = torch.randn(8,3)
-- input = torch.randn(10,1)
-- di1 = mod:backward(input, df1)
-- di2 = mod:backward(input_10_3, df2)


-- mod2 = nn.TemporalLogExpPooling(3,2,2)

-- input_7_3 = torch.randn(7,3)
-- input_9_5 = torch.randn(9,5)
-- out_7_3 = mod2:forward(input_7_3)
-- out_9_5 = mod2:forward(input_9_5)

-- df11 = torch.randn(3,3)
-- df22 = torch.randn(4,5)
-- di11 = mod2:backward(input_7_3, df11)
-- di22 = mod2:backward(input_9_5, df22)
