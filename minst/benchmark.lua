------------------------------------------------
--usage: th benchmark.lua
--output: forward time, backward time; using the model in model.lua and dataset in data.lua


opt={
    size='small'
}
dofile("data.lua")
dofile("model.lua")
dofile("loss.lua")
local t1 = sys.clock()
local cnt = 0
local fT = 0
local bT = 0
for t = 1,trainData:size() do
    model:zeroGradParameters()
    local input = trainData.data[t]:double()
    local target = trainData.labels[t]+1
    local c1 = sys.clock()
    local output = model:forward(input)
    local err = criterion:forward(output,target)
    local c2 = sys.clock()
    local df = criterion:backward(output, target)
    model:backward(input,df)
    local c3 = sys.clock()
    fT = fT+c2-c1
    bT = bT+c3-c2
    model:updateParameters(1e-3)
    local _, index = torch.max(output,1)
    if index[1]==target then cnt = cnt + 1 end
end
local t2 = sys.clock()
local time = t2-t1
print("train time per sample: "..(time*1000/trainData:size()).."ms")
print("forward time per sample: "..(fT*1000/trainData:size()).."ms")
print("backward time per sampel:"..(bT*1000/trainData:size()).."ms")
print("Accuracy: "..(1.0*cnt/trainData:size()*100).."%")
