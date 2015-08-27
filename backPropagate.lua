require 'torch'
require 'nn'
ninputs = 10
nhidden = ninputs/2
nhidden = 1
noutputs = 1

w1 = torch.rand(nhidden,ninputs)
b1 = torch.rand(nhidden)
d1 = torch.zeros(nhidden)

d2 = torch.zeros(nhidden)

w3 = torch.rand(noutputs,nhidden)
b3 = torch.rand(noutputs)
d3 = torch.zeros(noutputs)

input = torch.rand(ninputs)
res = torch.sum(input)
function forward()
    r1= torch.mv(w1, input)
    r1:add(b1)
    --print(r1)
    r2= torch.tanh(r1)
    --print(r2)
    r3= torch.mv(w3, r2)
    r3:add(b3)
    --print(r3)
end
function dtanh(x)
    ex = torch.exp(x)
    nex = torch.exp(-x)
    return 4/((ex+nex)*(ex+nex))
end

function backPropagate()
    d1:zero()
    d2:zero()
    d3:zero()

--[[
    r = r1[1]
    e = (r-res)*(r-res)
    d1[1] = 2*(r-res)
--]]

    r = r3[1]
    e = (r-res)*(r-res)
    --print(e)
    d3[1] = 2*(r-res)
    --print("===== d3 =====",d3)
    for i=1,(#d3)[1] do
        d2:add(torch.mul(w3,d3[i]))
    end
    --print("===== d2 =====", d2)
    d1 = r1:clone()
    d1:apply(dtanh)
    d1:cmul(d2)
    --d1 = d2
    --print("===== d1 =====", d1)
    --print(d1)
end

rate = 0.2
function updateW()
    rate = rate / 1.5
    if(rate < 0.001) then rate = 0.001 end
    dw3 = torch.Tensor(w3:size()):zero()
    for i=1,(#d3)[1] do
        dw3[i] = torch.mul(r2,d3[i])
    end
    db3 = d3
    w3:add(torch.mul(dw3,-rate))
    b3:add(torch.mul(db3,-rate))
    --print(dw3)
    --print(db3)

    --print("----- begin =====")
    --print("===== res =====")
    --print(res)
    --print("===== r =====")
    --print(r)
    --print("===== error =====")
    --print(e)
    --print("===== d1 =====")
    --print(d1)
    dw1 = torch.Tensor(w1:size()):zero()
    for i=1,(#d1)[1] do
        dw1[i] = torch.mul(input,d1[i])
    end
    db1 = d1
    w1:add(torch.mul(dw1,-rate))
    b1:add(torch.mul(db1,-rate))
--[[
    print("===== dw1 =====")
    print(dw1)
    print("===== db1 =====")
    print(db1)
    print("===== w1 =====")
    print(w1)
    print("===== b1 =====")
    print(b1)
    print("===== w1 af =====")
    print(w1)
    print("===== b1 af =====")
    print(b1)
   -- print(dw1)
    --print(db1)
--]]
end

function train(num)
    for i=1,num do
        input = torch.rand(ninputs)
        res = torch.sum(input)
        --res = input[1]
        forward()
        backPropagate()
        updateW()
        print("error:",e)
    end
    print(w1)
    print(b1)
    print(w3)
    print(b3)
end

train(1000)
