opt={
    size='small'
}
dofile("data.lua")
--dofile("1_data.lua")
dofile("model.lua")
dofile("loss.lua")
dofile("train.lua")
dofile("test.lua")
print("===train")
local i = 100
while true do
    if(i==0) then
        i=10
        os.execute("sleep 3")
    end
train()
test()
i=i-1
end
