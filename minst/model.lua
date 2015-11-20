require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

-- 14-class problem

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2


-- if opt.model == 'linear' then

   -- Simple linear model
--[[
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))
--]]

--elseif opt.model == 'mlp' then

-- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(1,28,28))
   model:add(nn.SpatialConvolution(1,32,5,5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(3,3,3,3))
   model:add(nn.SpatialConvolution(32,64,5,5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(2,2,2,2))
    ninputs = 64*2*2
    nhiddens = 200

   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))
--end

print(model)
