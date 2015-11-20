require 'torch'
loaded = torch.load("train.t7")
trsize = (#loaded.data)[1]
trsize = 5000
print("===train Data size:",trsize)
classes = loaded.className
trainData = {
    data = loaded.data:float(),
    labels = loaded.label,
    size = function() return trsize end
}
print(trainData)


loaded = torch.load("test.t7")
tesize = (#loaded.data)[1]
tesize = 1000
print("===test Data size:",tesize)
testData = {
    data = loaded.data:float(),
    labels = loaded.label,
    size = function() return tesize end
}
print(testData)

ninputs = (#trainData.data)[2] * (#trainData.data)[3]
noutputs = #classes
print("out",noutputs)
