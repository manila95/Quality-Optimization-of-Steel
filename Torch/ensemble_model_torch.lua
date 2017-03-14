require 'nn'
require "nngraph"
require 'optim'
matio = require 'matio'
require 'xlua'
require 'pl'
require 'paths'
require 'torch'
require 'math'
require "rnn"
-- require 'nnx'

opt = {}
opt.learning_rate = 0.01
opt.num_epochs = 10000
opt.save = "../logs"
opt.batch_size = 1
opt.momentum = 0.0
dropout = 0.5

trainset = matio.load("../Data/train_aim_qlty.mat")
testset = matio.load("../Data/test_aim_qlty.mat")

time = sys.clock()
trainLogger = optim.Logger(paths.concat("../" .. opt.save, 'train' .. tostring(time) .. '.log'))
-- valLogger = optim.Logger(paths.concat("./" .. opt.save .. opt.data, 'val.log'))
testLogger = optim.Logger(paths.concat("../" .. opt.save, 'test' .. tostring(time) .. '.log'))


-- mlp1
mlp1 = nn.Sequential()
mlp1:add(nn.Linear(14, 14))
mlp1:add(nn.ReLU())
-- mlp:add(nn.Droupout(opt.dropout))
mlp1:add(nn.Linear(14, 10))
mlp1:add(nn.ReLU())
mlp1:add(nn.Dropout(opt.dropout))
mlp1:add(nn.Linear(10, 5))
mlp1:add(nn.ReLU())
mlp1:add(nn.Dropout(opt.dropout))
mlp1:add(nn.Linear(5, 3))

-- mlp2
mlp2 = nn.Sequential()
mlp2:add(nn.Linear(14, 14))
mlp2:add(nn.ReLU())
-- mlp2:add(nn.Dropout(opt.dropout))
mlp2:add(nn.Linear(14, 10))
mlp2:add(nn.ReLU())
mlp2:add(nn.Dropout(opt.dropout))
mlp2:add(nn.Linear(10, 5))
mlp2:add(nn.ReLU())
mlp2:add(nn.Dropout(opt.dropout))
mlp2:add(nn.Linear(5, 3))

lt = nn.LookupTableMaskZero(3, 3)

rnn = nn.Sequential()
rnn:add(lt) -- will return a sequence-length x batch-size x embedDim tensor
rnn:add(nn.SplitTable(1, 3)) -- splits into a sequence-length table with batch-size x embedDim entries
rnn:add(nn.Sequencer(nn.LSTM(3, 20)))
rnn:add(nn.SelectTable(-1))
rnn:add(nn.Linear(20, 3)) -- map last state to a score for classification

net = nn.ParallelTable()
net:add(mlp1)
net:add(mlp2)
net:add(rnn)
-- net:add(nn.CAddTable())
model = nn.Sequential()
model:add(net)
model:add(nn.JoinTable(2))
model:add(nn.Reshape(9))
model:add(nn.Linear(9, 3))

criterion = nn.MSECriterion()

params, grad_params = model:getParameters()

function trainset:size()
    return trainset["chemical_parameters"]:size(1)
end
function testset:size()
    return testset["chemical_parameters"]:size(1)
end

function train(dataset)
    
    epoch = epoch or 1
    model:training()
    start_idx = 1
    for i = 1, 30000 do
        xlua.progress(i, dataset:size())
        local input1 = dataset["chemical_parameters"][{{i}}]
        local input2 = dataset["process_parameters"][{{i}}]
        local input3 = dataset["cooling_sequence"][{{i}}]:t()
        local targets = dataset["output_parameters"][{{i}}]
        
--         start_idx = end_idx + 1
        
--         print(inputs:size())
        
        local function feval(x)
            
             collectgarbage()
             if x ~= params then
                params:copy(x)
             end
             grad_params:zero()

             -- forward
--              print(inputs)
             local outputs = model:forward{input1, input2, input3}
             local err = criterion:forward(outputs, targets)
    
             -- backward
             local gradOutputs = criterion:backward(outputs, targets)
             model:zeroGradParameters()
             model:backward({input1, input2, input3}, gradOutputs)

            

--              -- gradient clipping
--              if opt.cutoff > 0 then
--                 local norm = rnn:gradParamClip(opt.cutoff) -- affects gradParams
--                 opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
--              end
--              print(err)
             return err, grad_params
        end
        
        adam_state = {
            learningRate = opt.learning_rate,
            momentum = opt.momentum
        }
        
        loss, _ = optim.adam(feval, params, adam_state)
        
    end
    model:evaluate()
    epoch = epoch + 1
end

function eval(dataset, targets)
    loss = 0
    for i = 1, 235 do 
        outputs = rnn:forward(dataset["chemical_parameters"][{{i}}], dataset["process_parameters"][{{i}}], dataset["cooling_sequence"][{{i}}]:t())
        loss = loss + criterion:forward(outputs, targets[{{i}}])
    end
--     rrmse = relative_rmse(outputs, targets)
--     mean_rrmse = torch.mean
--     result = {loss, rrmse[1][1], rrmse[1][2], rrmse[1][3]}
    return loss/235.0
end

function relative_rmse(outputs, targets)
    
    mean_ = torch.mean(targets, 1)
    squared_error = torch.sum(torch.pow(outputs - targets, 2), 1)
--     print(squared_error)
    mse = torch.div(squared_error, targets:size(1))
    rmse = torch.sqrt(mse)
    rrmse = torch.cdiv(rmse, mean_)
    return rrmse
end

print("Epoch" .. "         " .. "Loss " .. "         " .. "EL RRMSE" .. "         " .. "UTS RRMSE" .. "         " .. "LYS RRMSE")

best_loss = 1900
for i = 1, 1000 do 
    train(trainset)
    if i % 1 == 0 then
        result = eval(testset, testset["output_parameters"])
        if result[1] < best_loss then
            best_loss = result[1]
            torch.save("../checkpoints/parallel_mlp_rnn_sgd.t7", model)
        end
        print(tostring(i) .. "         " .. tostring(result[1]) .. "         " .. tostring(result[2]) .. "         " .. tostring(result[3]) .. "         " .. tostring(result[4]))
    end
end

