import json
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from random import randint

infile=open("./doc_scm.json","r",encoding="utf-8")
# tmp = []
# for line in infile:
#     tmp.append(json.loads(line))
#
# print(len(tmp))
# print(tmp[0].keys())
alldata=json.load(infile)
keys=list(alldata.keys())
num=len(keys)
tokenizer=BertTokenizer.from_pretrained('/data/niuzuoyao/Bert-Chinese-Text-Classification-Pytorch-master/bert_pretrain')
model=BertModel.from_pretrained('/data/niuzuoyao/Bert-Chinese-Text-Classification-Pytorch-master/bert_pretrain')

# 将resnet架构迁移到设备
model.to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# str="张三犯毒品罪"
# token=tokenizer.tokenize(str)
# token_ids=torch.tensor([tokenizer.convert_tokens_to_ids(token)]).to('cuda')
# _,query=model(token_ids)
# print(query.shape)

τ = 0.05

def loss_function(query, pos, neg):

    pos = torch.exp(torch.div(torch.cosine_similarity(query,pos),τ))

    # 在查询和队列张量之间执行矩阵乘法
    neg = torch.exp(torch.div(torch.cosine_similarity(query,neg),τ))

    # 求和
    denominator = neg + pos

    return -torch.log(torch.div(pos,denominator))


def train(step):
    model.eval()
    for i in range(step):
        query_field=alldata[keys[randint(0,num-1)]]
        query=query_field[randint(0,len(query_field)-1)]

        pos=query_field[randint(0,len(query_field)-1)]

        neg_field=alldata[keys[randint(0,num-1)]]
        neg=neg_field[randint(0,len(neg_field)-1)]

        query_token=tokenizer.tokenize(query)
        pos_token=tokenizer.tokenize(pos)
        neg_token=tokenizer.tokenize(neg)
        
        if len(query_token)>=512:
            query_token=query_token[:512]
        if len(pos_token)>=512:
            pos_token=pos_token[:512]
        if len(neg_token)>=512:
            neg_token=neg_token[:512]


        query_ids=torch.tensor([tokenizer.convert_tokens_to_ids(query_token)]).to('cuda')
        pos_ids=torch.tensor([tokenizer.convert_tokens_to_ids(pos_token)]).to('cuda')
        neg_ids=torch.tensor([tokenizer.convert_tokens_to_ids(neg_token)]).to('cuda')

        # 梯度零化
        optimizer.zero_grad()


        # 获取他们的输出
        _,query_en = model(query_ids)
        _,pos_en = model(pos_ids)
        _,neg_en = model(neg_ids)

        # 获得损失值
        loss = loss_function(query_en,pos_en,neg_en)

        # 反向传播
        loss.backward()

        # 运行优化器
        optimizer.step()

        if i % 10==0:
            print("step {}:{}".format(i/10,loss.item()))

train(10000)
