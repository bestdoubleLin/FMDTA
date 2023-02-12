import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gep, global_max_pool as gmp
from torch_geometric.nn import GCNConv

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=54,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()
        # d_model = 128  # Embedding Size
        # d_ff = 512  # FeedForward dimension
        # d_k = d_v = 64  # dimension of K(=Q), V
        # n_layers = 6  # number of Encoder of Decoder Layer
        # n_heads = 8  # number of heads in Multi-Head Attention
        dmax_length=100
        dv_size=62+1
        tmax_length = 850
        tv_size = 26

        # # 以下是药的GIN
        # dim = 32
        # self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        # self.n_output = n_output
        # # convolution layers
        # nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        # self.conv1 = GINConv(nn1)
        # self.bn1 = torch.nn.BatchNorm1d(dim)
        #
        # nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv2 = GINConv(nn2)
        # self.bn2 = torch.nn.BatchNorm1d(dim)
        #
        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)
        #
        # nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv4 = GINConv(nn4)
        # self.bn4 = torch.nn.BatchNorm1d(dim)
        #
        # nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv5 = GINConv(nn5)
        # self.bn5 = torch.nn.BatchNorm1d(dim)
        #
        # self.fc1_xd = Linear(dim, output_dim)

        # 以下是药的GATGCN
        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        output_dim2 = 128
        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_xt, num_features_xt)
        self.pro_conv2 = GCNConv(num_features_xt, num_features_xt * 2)
        self.pro_conv3 = GCNConv(num_features_xt * 2, num_features_xt * 4)
        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_xt * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim2)


        # 1D convolution on drug sequence
        # self.embedding_smiles = nn.Embedding(62 + 1, embed_dim)    # (词典的大小尺寸,嵌入向量的维度)
        self.drug = Transformer(dmax_length,dv_size)
        self.conv_smi_1 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=8)
        self.fc1_smi = nn.Linear(32*121, output_dim)
        # self.fc1_smi = nn.Linear(100*128, output_dim)

        # 1D convolution on protein sequence
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.target = Transformer(tmax_length, tv_size)
        self.conv_xt_1 = nn.Conv1d(in_channels= 850, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)
        # self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # self.fc1_xt = nn.Linear(1000 * 128, output_dim)

        self.fc4 = nn.Linear(output_dim*2,output_dim)
        self.fc5 = nn.Linear(output_dim + output_dim2, output_dim)

        #对比
        self.constrast = SupConLoss()

        #我的
        self.fc1 = nn.Linear(output_dim * 3 + output_dim2, 1024)
        # self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)


    def forward(self, data_mol,data_pro):

        # x, edge_index, batch = data_mol.x.to(device), data_mol.edge_index.to(device), data_mol.batch.to(device)
        # target_x, target_edge_index, target_batch = data_pro.x.to(device), data_pro.edge_index.to(device), data_pro.batch.to(device)
        # smi = data_mol.drug_smiles.to(device)
        # target = data_pro.target.to(device)
        x, edge_index, batch = data_mol.x, data_mol.edge_index, data_mol.batch
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch
        smi = data_mol.drug_smiles
        target = data_pro.target

        # drug embedding
        #以下是药的GIN
        # x = F.relu(self.conv1(x, edge_index))
        # x = self.bn1(x)
        # x = F.relu(self.conv2(x, edge_index))
        # x = self.bn2(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)
        # x = F.relu(self.conv5(x, edge_index))
        # x = self.bn5(x)
        # x = global_add_pool(x, batch)
        # x = F.relu(self.fc1_xd(x))
        # x = F.dropout(x, p=0.2, training=self.training)

        #以下是药的GAT
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gep(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)
        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)
        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)
        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling
        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)


        # # drug smiles embedding
        # embedding_smiles = self.embedding_smiles(smi)
        # conv_xd = self.conv_smi_1(embedding_smiles)
        # # flatten
        # embedded_smiles = conv_xd.view(-1, 32 * 121)
        # embedded_smiles = self.fc1_smi(embedded_smiles)
        #
        # # concat drug smiles and drug graph
        # xd = torch.cat((x, embedded_smiles), 1)
        embedding_smiles=self.drug(smi)
        # embedded_smiles = embedding_smiles.view(-1, 100*128)
        # embedded_smiles = self.fc1_smi(embedded_smiles)
        conv_xd = self.conv_smi_1(embedding_smiles)
        # # flatten
        embedded_smiles = conv_xd.view(-1, 32 * 121)
        embedded_smiles = self.fc1_smi(embedded_smiles)

        drug_con = self.constrast(x,embedded_smiles,labels=None, mask=None)

        # print(embedded_smiles.shape)
        # # concat drug smiles and drug graph
        xd = torch.cat((x, embedded_smiles), 1)
        # xd = self.fc4(xd)

        # target embedding
        # embedded_xt = self.embedding_xt(target)
        embedded_xt = self.target(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xp = conv_xt.view(-1, 32 * 121)
        xp = self.fc1_xt(xp)
        # xp = embedded_xt.view(-1, 400 * 128)
        # xp = self.fc1_xt(xp)

        pro_con = self.constrast(xt, xp, labels=None, mask=None)

        con = drug_con + pro_con

        xpp = torch.cat((xt, xp), 1)
        # xpp = self.fc5(xpp)

        # concat drug and target
        xc = torch.cat((xd, xpp), 1)
        # concat
        # xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out,con


## 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
class Transformer(nn.Module):
    def __init__(self,max_length,v_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(max_length,v_size)  ## 编码层
        # self.decoder = Decoder()  ## 解码层
        # self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
    def forward(self, enc_inputs):
        ## 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码段的输入，一个dec_inputs，形状为[batch_size, tgt_len]，主要是作为解码端的输入

        ## enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        ## enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # enc_outputs = self.encoder(enc_inputs)
        # ## dec_outputs 是decoder主要输出，用于后续的linear映射； dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        # dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        #
        # ## dec_outputs做映射到词表大小
        # dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        # return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
        return enc_outputs
## 2. Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self,max_len,src_vocab_size,d_model=128,n_layers=2):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  ## 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(d_model,max_len) ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, enc_inputs):
        ## 这里我们的 enc_inputs 形状是： [batch_size x source_len]

        ## 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)

        ## 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现；3.
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        ##get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
## 3. PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len,dropout=0.1):
        super(PositionalEncoding, self).__init__()

        ## 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        ## pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        ##假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

## 4. get_attn_pad_mask

## 比如说，我现在的句子长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
## len_input * len*input  代表每个单词对其余包含自己的单词的影响力

## 所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算计算softmax之前会把这里置为无穷大；

## 一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要

## seq_q 和 seq_k 不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的；

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

## 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=128,d_k=32,d_v=32,n_heads=2):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask,d_k=32,d_v=32,n_heads=2):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]
## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k=16):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model=128,d_ff=32):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features1,features2, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features1.is_cuda
                  else torch.device('cpu'))
        # features1 = F.normalize(features1, p=2, dim=1)
        # features2 = F.normalize(features1, p=2, dim=1)
        # batch_size = features1.shape[0]

        features = torch.cat((features1,features2),0)
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        batch = features1.shape[0]

        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch, dtype=torch.float32).to(features.device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        # mask = 2 * mask - torch.ones_like(mask)
        mask = torch.cat((mask,mask), 1)
        mask = torch.cat((mask, mask), 0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast
        exp_logits = torch.exp(logits)

        # 构建mask
        logits_mask = torch.ones_like(mask).to(exp_logits.device) - torch.eye(batch_size).to(exp_logits.device)
        positives_mask = (mask * logits_mask).to(logits_mask.device)
        negatives_mask = (1. - mask).to(logits_mask.device)

        num_positives_per_row = torch.sum(positives_mask, axis=1).to(logits_mask.device)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        # log_probs = torch.sum(
        #     (logits - torch.log(denominator)) * positives_mask, axis=1) / 1

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss