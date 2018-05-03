import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding, QuestionEmbedding_all
from classifier import SimpleClassifier
from gcn import GraphConvolution
from fc import FCNet
import pdb
from torch.autograd import Variable

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        # pdb.set_trace()
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class GraphModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net0, v_net0, gcn, q_net, v_net, classifier):
        super(GraphModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net0 = q_net0
        self.v_net0 = v_net0
        self.gcn = gcn       
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        # pdb.set_trace()
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        q_rep = self.q_net0(q_emb)
        v_rep = self.v_net0(v)

        # compute affinity matrix
        v_rep_p = v_rep.permute(0, 2, 1)
        qv_aff = torch.bmm(q_rep, v_rep_p)
        vq_aff = qv_aff.permute(0, 2, 1)
        qq_aff = torch.bmm(qv_aff, vq_aff)
        vv_aff = torch.bmm(vq_aff, qv_aff)

        qq_aff = Variable(qq_aff.data.fill_(0))
        vv_aff = Variable(vv_aff.data.fill_(0))
        
        # compose a single graph with [aff, aff_p, q_aff, v_aff]        
        q_aff = torch.cat((qq_aff, qv_aff), 2)
        v_aff = torch.cat((vq_aff, vv_aff), 2)
        aff = torch.cat((q_aff, v_aff), 1)

        # perform gcn on question feature based on q_adj
        comb_rep = torch.cat((q_rep, v_rep), 1)
        comb_rep = self.gcn(comb_rep)

        gcn_rep = comb_rep + torch.bmm(aff, comb_rep)
        joint_repr = gcn_rep[:, 14:].mean(1)

        # pdb.set_trace()
        # att = self.v_att(v, q_emb)
        # v_emb = (att * v).sum(1) # [batch, v_dim]
        # q_repr = self.q_net(q_emb)
        # v_repr = self.v_net(v_emb)
        # joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)
        return logits

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    # pdb.set_trace()
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_baseline0_gcn(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding_all(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)

    q_net0 = FCNet([q_emb.num_hid, num_hid])
    v_net0 = FCNet([dataset.v_dim, num_hid])

    gcn = FCNet([num_hid, num_hid])

    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return GraphModel(w_emb, q_emb, v_att, q_net0, v_net0, gcn, q_net, v_net, classifier)

