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
local i = 10
while i>0 do
        if cmd == "n" then break end
    train()
    test()
    i=i-1
    if(i==0) then
        io.write("more loops [number]:")
        i = io.read("*number")
        if not i then break end
    end
end
