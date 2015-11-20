-------------------------------------------------------------
--Usage: th createDataFile.lua dataPath labelPaht train/test
--Description: convert the binary mnist data file into torch7 format, the binary file can be downloaded from 
--minst: http://yann.lecun.com/exdb/mnist/

require 'torch'
local tablex = require 'pl.tablex'

-- minst: http://yann.lecun.com/exdb/mnist/
arg = arg or {}
local dataPath = arg[1] or 'dataset/t10k-images-idx3-ubyte'
local labelPath = arg[2] or 'dataset/t10k-labels-idx1-ubyte'
local filename = arg[3] or "test"
if not dataPath or not labelPath then
    print("usage:- dataPath labelPath train/test")
    return
end
if filename~="test" and filename~="train" then
    print("output must be test or train")
    print("usage:- dataPath labelPath train/test")
    return
end

function readInt(f, len)
    len = len or 4
    local bytes = f:read(len)
    local res = 0
    for i=1,len do
        res = 256*res + bytes:byte(i)
    end
    return res
end

function readIdxFile(filename)
    local dataFile = io.open(filename, 'rb')
    readInt(dataFile,2)
    local bits = readInt(dataFile,1)
    local dims = readInt(dataFile,1)
    local size = torch.LongStorage(dims)
    local total = 1
    for i=1,dims do
        size[i] = readInt(dataFile,4)
        print(size[i])
        total = total * size[i]
    end

    local dataBytes = dataFile:read(total)
    local dataTensor = torch.Tensor(size)
    local i = 0
    dataTensor:apply(function(x) i = i+1; return dataBytes:byte(i) end)
    dataFile:close()
    return dataTensor
end
local className = {'0','1','2','3','4','5','6','7','8','9'}
object = {
    data = readIdxFile(dataPath),
    label = readIdxFile(labelPath),
    className = className,
    name = "minst data"
}

torch.save(filename..".t7", object)
print("success! file saved as "..filename..".t7")

