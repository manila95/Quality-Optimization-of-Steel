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

opt = lapp[[
   -s,--save          (default "logs/")              subdirectory to save logs
   -p,--plot                                         plot while training
   -o,--optimization  (default "Adam")               optimization: SGD | LBFGS | Adam
   -l,--learningRate  (default 0.1)                  learning rate
   -b,--batchSize     (default 10)                   batch size
   -m,--momentum      (default 0)                    momentum, for SGD only
   -i,--maxIter       (default 1000)                 maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)                    L1 penalty on the weights
   --coefL2           (default 0)                    L2 penalty on the weights
   -t,--type          (default "cpu")                GPU or CPU
]]

-- Load dataset 
trainset = matio.load("../Data/train_aim_qlty" .. "I09499" .. ".mat")
testset = matio.load("../Data/test_aim_qlty" .. "I09499" .. ".mat")

function trainset:size()
    return trainset["chemical_parameters"]:size(1)
end
function testset:size()
    return testset["chemical_parameters"]:size(1)
end

lt = nn.LookupTableMaskZero(3, 3)

rnn = nn.Sequential()
rnn:add(lt) 
rnn:add(nn.SplitTable(1, 3)) 
rnn:add(nn.Sequencer(nn.LSTM(3, 3)))
rnn:add(nn.SelectTable(-1)) -- selects last state of the LSTM
rnn:add(nn.Linear(3, 3)) 


criterion = nn.MSECriterion()
params, grad_params = rnn:getParameters()

function train(dataset)
    
    epoch = epoch or 1
    
    start_idx = 1
    for end_idx = opt.batch_size, dataset:size(), opt.batch_size do
        
        local inputs = dataset["cooling_sequence"][{{start_idx, end_idx}}]:t()
        local targets = dataset["output_parameters"][{{start_idx, end_idx}}]
        
        start_idx = end_idx + 1

        
        local function feval(x)
            
             collectgarbage()
             if x ~= params then
                params:copy(x)
             end
             grad_params:zero()

             -- forward
--              print(inputs)
             local outputs = rnn:forward(inputs)
             local err = criterion:forward(outputs, targets)
    
             -- backward
             local gradOutputs = criterion:backward(outputs, targets)
             rnn:zeroGradParameters()
             rnn:backward(inputs, gradOutputs)

            

             -- gradient clipping
             if opt.cutoff > 0 then
                local norm = rnn:gradParamClip(opt.cutoff) -- affects gradParams
                opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
             end
            
             return err, grad_params
        end
        
        adam_state = {
            learningRate = opt.learning_rate,
            momentum = opt.momentum
        }
        
        loss, _ = optim.adam(feval, params, adam_state)
        
    end
    
    epoch = epoch + 1
end
    
function eval(data, targets)
    
    local outputs = rnn:forward(data)
    local loss = criterion:forward(outputs, targets)
    return loss
end

for i = 1, 4 do 
    train(trainset)
    if i % 2 == 0 then
        print(eval(testset["cooling_sequence"]:t(), testset["output_parameters"]))
    end
end