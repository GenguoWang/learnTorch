require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

opt = {
    save="results",
    optimization="SGD",
    learningRate=1e-3,
    batchSize=1,
    weightDecay=0,
    momentum=0,
    t0=l,
    maxIter=2
}

confusion = optim.ConfusionMatrix(classes)
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if model then
   parameters,gradParameters = model:getParameters()
end

print '==> configuring optimizer'
--use sgd
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = 1e-7
}
optimMethod = optim.sgd

print '==> defining training procedure'
function train()
    epoch = epoch or 1
    local time = sys.clock()
    model:training()
    shuffle = torch.randperm(trsize)
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,trainData:size(),opt.batchSize do
        xlua.progress(t, trainData:size())
        local inputs = {}
        local targets = {}
        for i = t, math.min(t+opt.batchSize-1,trainData:size()) do
            local input = trainData.data[shuffle[i]]
            local target = trainData.labels[shuffle[i]]
            input = input:double()
            table.insert(inputs,input)
            table.insert(targets, target)
        end
        local feval = function(x)
                       if x ~= parameters then
                          parameters:copy(x)
                       end
                       gradParameters:zero()
                       local f = 0
                       for i = 1,#inputs do
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          confusion:add(output, targets[i])
                       end
                       gradParameters:div(#inputs)
                       f = f/#inputs
                       return f,gradParameters
                    end
        optimMethod(feval, parameters, optimState)
    end
    time = sys.clock() - time
    time = time / trainData:size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print(confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    torch.save(filename, model)
    confusion:zero()
    epoch = epoch + 1
end
