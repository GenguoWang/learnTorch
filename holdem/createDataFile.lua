local path = "/home/kingo/workshop/dataset/test"

require 'torch'
require 'image'
local dir = require 'pl.dir'
local dirs = dir.getdirectories(path);
local classes = {}
local className = {}
local data = {}
local idx = 0
for k,dirpath in ipairs(dirs) do
    local class = paths.basename(dirpath)
    print(class)
    table.insert(className,class)
    idx = idx + 1
    print(dirpath)
    local imgFiles = dir.getfiles(dirpath)
    print(imgFiles)
    for k,imgFile in ipairs(imgFiles) do
        print(imgFile)
        table.insert(classes, idx)
        table.insert(classes, idx)
        table.insert(data, image.load(imgFile))
        table.insert(data, image.load(imgFile))
    end
end
classTs = torch.LongTensor(#classes)

oneData = data[1]
oneSize = #oneData
dataSize = torch.LongStorage(#oneSize+1)
dataSize[1] = #data
for i = 1, #oneSize do dataSize[i+1] = oneSize[i] end
print(dataSize)
dataTs = torch.Tensor(dataSize):type(oneData:type())
for i = 1, #classes do
    classTs[i] = classes[i]
    dataTs[{{i}}]:copy(data[i])
end
print(classTs)
print(#dataTs)

object = {
    data = dataTs,
    label = classTs,
    labelName = className,
    name = "kingo data"
}
torch.save("train.t7", object)
print("success! file saved as train.t7")
