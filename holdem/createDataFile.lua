local tablex = require 'pl.tablex'

classes = {'2','3','4','5','6','7','8','9','T','J','Q','K','A','B'}

local path = arg[1]
local filename = arg[2] or "train"
if not path then
    print("usage:- path")
    return
end


require 'torch'
require 'image'
local dir = require 'pl.dir'
local dirs = dir.getdirectories(path);
local className = {'2','3','4','5','6','7','8','9','T','J','Q','K','A','B'}
local classes = {}
local data = {}
for k,dirpath in ipairs(dirs) do
    local class = paths.basename(dirpath)
    print(class)
    table.insert(className,class)
    local idx = tablex.find(className,class)
    print(dirpath)
    local imgFiles = dir.getfiles(dirpath)
    print(imgFiles)
    for k,imgFile in ipairs(imgFiles) do
        print(imgFile)
        table.insert(classes, idx)
        table.insert(data, image.load(imgFile))
    end
end
classTs = torch.LongTensor(#classes)

oneData = data[1]
oneSize = #oneData
dataSize = #oneData
dataSize[1] = #data
--[[
dataSize = torch.LongStorage(#oneSize+1)
dataSize[1] = #data
for i = 1, #oneSize do dataSize[i+1] = oneSize[i] end
print(dataSize)
--]]
dataTs = torch.Tensor(dataSize):type(oneData:type())
print(#dataTs)
for i = 1, #classes do
    classTs[i] = classes[i]
    dataTs[{{i}}]:copy(data[i][1])
end
print(classTs)
print(#dataTs)

object = {
    data = dataTs,
    label = classTs,
    labelName = className,
    name = "kingo data"
}
torch.save(filename..".t7", object)
print("success! file saved as "..filename..".t7")
