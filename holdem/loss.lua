require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

--if opt.loss == 'margin' then
    --criterion = nn.MultiMarginCriterion()

--elseif opt.loss == 'nll' then
   model:add(nn.LogSoftMax())
   criterion = nn.ClassNLLCriterion()

--elseif opt.loss == 'mse' then
    --model:add(nn.Tanh())
print(criterion)
