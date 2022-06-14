from net import *
from mydataset import *
from paddlenlp.transformers import GPTModel, GPTTokenizer
tokenizer = GPTTokenizer.from_pretrained("./token")
train_path = "./data/train.pkl"
tran_data = MyDataset(train_path)

import paddle
from paddlenlp.ops.optimizer import AdamWDL

# linear = paddle.nn.Linear(10, 10)
print(tokenizer.vocab_size)
print(tokenizer.sep_token_id)


net = MyModel(tokenizer.vocab_size+1)
name_dict = dict()



adam = paddle.optimizer.Adam(parameters=net.parameters(),learning_rate=0.001)


batch_size = 10
for epoch in range(10):
    for i,data in enumerate(tran_data):
        content,label,lenght = data
        out = net(content)
        out = out[:,lenght:,:]
        loss =paddle.nn.functional.cross_entropy(out,label)
        print(f"epoch:{epoch} step:{i} loss:{loss.item()}")
        loss.backward()
        if i%batch_size==0:
            adam.step()
            adam.clear_grad()
        if i%500==0:
            paddle.save(net.state_dict(),f'model/model_{epoch}_{i}.pkl')

