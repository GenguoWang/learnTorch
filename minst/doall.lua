--------------------------------------------
--train and test 
--------------------------------------------

opt={
    size='small'
}
dofile("data.lua")
dofile("model.lua")
dofile("loss.lua")
dofile("train.lua")
dofile("test.lua")
print("===train")
local i = 10
while i>0 do
    train()
    test()
    i=i-1
    if(i==0) then
        io.write("more loops [number]:")
        i = io.read("*number")
        if not i then break end
    end
end
