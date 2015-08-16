require 'torch'
loaded = torch.load("train.t7")
trsize = (#loaded.data)[1]
print("===train Data size:",trsize)
trainData = {
    data = loaded.data:float(),
    labels = loaded.label,
    size = function() return trsize end
}
print(trainData)


loaded = torch.load("test.t7")
tesize = (#loaded.data)[1]
testData = {
    data = loaded.data:float(),
    labels = loaded.label,
    size = function() return tesize end
}
print(testData)

noutputs = 14
ninputs = 12*18
classes = {'2','3','4','5','6','7','8','9','T','J','Q','K','A','B'}
