import torch
from torch_geometric.nn.models import SignedGCN
import torch.nn.functional as F


class SGNNEnc(torch.nn.Module):
    """Learn the embeddings for each node"""

    def __init__(self, usr_num, qus_num, args):
        super(SGNNEnc, self).__init__()

        self.args = args
        self.emb_size = args.emb_size  # embedding size
        self.emb_out = None  # output embedding
        self.usr_num, self.qus_num = usr_num, qus_num  # user and question number

        # linear transformation
        self.linear_g1 = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)  # graph 1
        self.linear_g2 = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)  # graph 2
        self.linear_g3 = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)  # graph 3
        self.linear_g4 = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)  # graph 4

        # signed convolution layers
        self.sconv_p1 = SignedGCN(self.emb_size, self.emb_size, num_layers=args.num_layers)  # perspective 1
        self.sconv_p2 = SignedGCN(self.emb_size, self.emb_size, num_layers=args.num_layers)  # perspective 2

        # transform the concatenated embeddings from 4 graphs
        self.linear_combine = torch.nn.Linear(4 * self.emb_size, self.emb_size, bias=False)
        self.dropout = torch.nn.Dropout(p=args.dropout)

        # link MLP
        self.link_mlp = LinkMLP(args)

    def forward(self, x: torch.Tensor, edges: tuple, pos_neg_masks: tuple):
        # get edge index
        edge_index_g1, edge_index_g2, \
            edge_index_g3_u, edge_index_g3_q, \
            edge_index_g4_u, edge_index_g4_q = edges  # Tensor

        # get positive and negative edge index
        mask_g1, mask_g2, mask_g3_u, mask_g3_q, mask_g4_u, mask_g4_q = pos_neg_masks  # boolean array
        edge_index_g1_pos, edge_index_g1_neg = edge_index_g1[:, mask_g1], edge_index_g1[:, ~mask_g1]
        edge_index_g2_pos, edge_index_g2_neg = edge_index_g2[:, mask_g2], edge_index_g2[:, ~mask_g2]
        edge_index_g3_u_pos, edge_index_g3_u_neg = edge_index_g3_u[:, mask_g3_u], edge_index_g3_u[:, ~mask_g3_u]
        edge_index_g3_q_pos, edge_index_g3_q_neg = edge_index_g3_q[:, mask_g3_q], edge_index_g3_q[:, ~mask_g3_q]
        edge_index_g4_u_pos, edge_index_g4_u_neg = edge_index_g4_u[:, mask_g4_u], edge_index_g4_u[:, ~mask_g4_u]
        edge_index_g4_q_pos, edge_index_g4_q_neg = edge_index_g4_q[:, mask_g4_q], edge_index_g4_q[:, ~mask_g4_q]

        # linear transformation of the embeddings
        emb_g1 = self.linear_g1(x)  # graph 1
        emb_g2 = self.linear_g2(x)  # graph 2
        emb_g3_u = self.linear_g3(x)  # graph 3, user
        emb_g3_q = self.linear_g3(x)  # graph 3, question
        emb_g4_u = self.linear_g4(x)  # graph 4, user
        emb_g4_q = self.linear_g4(x)  # graph 4, question

        # graph convolution
        emb_g1 = self.sconv_p1(emb_g1, edge_index_g1_pos, edge_index_g1_neg)  # perspective 1
        emb_g2 = self.sconv_p1(emb_g2, edge_index_g2_pos, edge_index_g2_neg)

        emb_g3_u = self.sconv_p2(emb_g3_u, edge_index_g3_u_pos, edge_index_g3_u_neg)  # perspective 2
        emb_g3_q = self.sconv_p2(emb_g3_q, edge_index_g3_q_pos, edge_index_g3_q_neg)
        emb_g4_u = self.sconv_p2(emb_g4_u, edge_index_g4_u_pos, edge_index_g4_u_neg)
        emb_g4_q = self.sconv_p2(emb_g4_q, edge_index_g4_q_pos, edge_index_g4_q_neg)

        # combine Graph 3 and Graph 4 embeddings
        emb_g3 = torch.cat([emb_g3_u[:self.usr_num], emb_g3_q[self.usr_num:]], dim=0)
        emb_g4 = torch.cat([emb_g4_u[:self.usr_num], emb_g4_q[self.usr_num:]], dim=0)

        # drop out
        emb_g1 = self.dropout(emb_g1)
        emb_g2 = self.dropout(emb_g2)
        emb_g3 = self.dropout(emb_g3)
        emb_g4 = self.dropout(emb_g4)

        return emb_g1, emb_g2, emb_g3, emb_g4

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.linear_g1.reset_parameters()
        self.linear_g2.reset_parameters()
        self.linear_g3.reset_parameters()
        self.linear_g4.reset_parameters()
        self.sconv_p1.reset_parameters()
        self.sconv_p2.reset_parameters()
        self.linear_combine.reset_parameters()
        self.link_mlp.reset_parameters()

    def compute_contrastive_loss(self, x):
        def inter_contrastive(emb_1, emb_2):
            pos_score = torch.div(torch.bmm(emb_1.view(emb_1.shape[0], 1, emb_1.shape[1]),
                                            emb_2.view(emb_2.shape[0], emb_2.shape[1], 1)),
                                  self.args.tau).exp()
            neg_sim = torch.mm(emb_1, emb_2.transpose(0, 1)).fill_diagonal_(0)
            neg_score = torch.div(neg_sim, self.args.tau).exp().sum(dim=1)
            return (-(torch.div(pos_score, neg_score).log())).mean()

        def intra_contrastive(emb, emb_1, emb_2):
            score_1 = torch.bmm(emb.view(emb.shape[0], 1, emb.shape[1]),
                                emb_1.view(emb_1.shape[0], emb_1.shape[1], 1)).div(self.args.tau).exp()
            score_2 = torch.bmm(emb.view(emb.shape[0], 1, emb.shape[1]),
                                emb_2.view(emb_2.shape[0], emb_2.shape[1], 1)).div(self.args.tau).exp()
            return (-(torch.div(score_1, score_2).log())).mean()

        emb_g1, emb_g2, emb_g3, emb_g4 = x  # get the embeddings

        emb_g1 = F.normalize(emb_g1, p=2, dim=1)  # normalize the embeddings (l2)
        emb_g2 = F.normalize(emb_g2, p=2, dim=1)
        emb_g3 = F.normalize(emb_g3, p=2, dim=1)
        emb_g4 = F.normalize(emb_g4, p=2, dim=1)

        inter_loss_g12 = inter_contrastive(emb_g1, emb_g2)  # inter contrastive loss
        inter_loss_g34 = inter_contrastive(emb_g3, emb_g4)

        # get the combined embedding
        self.emb_out = self.linear_combine(torch.cat([emb_g1, emb_g2, emb_g3, emb_g4], dim=1))
        self.emb_out = F.normalize(self.emb_out, p=2, dim=1)

        intra_loss_p1 = intra_contrastive(self.emb_out, emb_g1, emb_g2)  # intra contrastive loss
        intra_loss_p2 = intra_contrastive(self.emb_out, emb_g3, emb_g4)

        return (1 - self.args.alpha) * (inter_loss_g12 + inter_loss_g34) + \
            self.args.alpha * (intra_loss_p1 + intra_loss_p2)  # sum up 2 contrastive losses

    def compute_label_loss(self, y_score, y_label):
        """Compute the label loss of the model using prediction and actual labels"""
        # positive sample weight (neg sample num / pos sample num)
        pos_weight = torch.Tensor([((y_label == 0).sum() / (y_label == 1).sum()).item()] * len(y_label)).to(
            y_score.device)
        return F.binary_cross_entropy_with_logits(y_score, y_label, pos_weight=pos_weight)

    def predict_edges(self, embeddings, uid, qid):
        """Predict the sign of edges given embeddings and user id, question id"""
        user_embedding = embeddings[uid]
        qust_embedding = embeddings[qid]
        return self.link_mlp(user_embedding, qust_embedding)


class LinkMLP(torch.nn.Module):
    """Predict the sign of edges using the learned embeddings"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.linear_predictor_layers == 0:  # use dot product
            pass
        elif args.linear_predictor_layers == 1:
            self.predictor = torch.nn.Linear(2 * args.emb_size, 1)
        elif args.linear_predictor_layers == 2:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(2 * args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, 1))
        elif args.linear_predictor_layers == 3:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(2 * args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, 1))
        elif args.linear_predictor_layers == 4:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(2 * args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, 1))
        else:
            raise Exception("Invalid layer number.")

    def forward(self, v_user: torch.Tensor, v_qust: torch.Tensor):
        if self.args.linear_predictor_layers == 0:  # dot product
            return v_user.mul(v_qust).sum(dim=-1)
        x = torch.cat([v_user, v_qust], dim=-1)  # concat the user and question embeddings
        return self.predictor(x).flatten()

    def reset_parameters(self):
        if hasattr(self, 'predictor'):
            self.predictor.reset_parameters()
