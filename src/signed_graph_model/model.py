import torch
from torch_geometric.nn import GATConv, SignedGCN
import torch.nn.functional as F


class GAT_CL(torch.nn.Module):
    """Use contrastive learning to get an embedding for each node"""

    def __init__(self, args, device):
        super(GAT_CL, self).__init__()
        self.args = args
        self.device = device
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.norm_embs = None  # normalized embeddings

        self.layer_ab_pos = torch.nn.ModuleList([GATConv(self.emb_size, self.emb_size) for _ in range(self.num_layers)])
        self.layer_ab_neg = torch.nn.ModuleList([GATConv(self.emb_size, self.emb_size) for _ in range(self.num_layers)])
        self.linear_combine = torch.nn.Linear(4 * self.emb_size, self.emb_size, bias=False)
        self.activation = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=args.dropout)
        self.link_mlp = LinkMLP(args)  # make prediction

    def forward(self, x, edge_index_g1_pos, edge_index_g2_pos, edge_index_g1_neg, edge_index_g2_neg):
        emb_g1_pos = x
        emb_g2_pos = x
        emb_g1_neg = x
        emb_g2_neg = x

        # positive edges
        for layer in self.layer_ab_pos:
            # graph 1
            emb_g1_pos = layer(emb_g1_pos, edge_index_g1_pos)
            emb_g1_pos = self.activation(emb_g1_pos)
            # graph 2
            emb_g2_pos = layer(emb_g2_pos, edge_index_g2_pos)
            emb_g2_pos = self.activation(emb_g2_pos)

        # negative edges
        for layer in self.layer_ab_neg:
            # graph 1
            emb_g1_neg = layer(emb_g1_neg, edge_index_g1_neg)
            emb_g1_neg = self.activation(emb_g1_neg)
            # graph 2
            emb_g2_neg = layer(emb_g2_neg, edge_index_g2_neg)
            emb_g2_neg = self.activation(emb_g2_neg)

        # dropout
        emb_g1_pos, emb_g2_pos = self.dropout(emb_g1_pos), self.dropout(emb_g2_pos)
        emb_g1_neg, emb_g2_neg = self.dropout(emb_g1_neg), self.dropout(emb_g2_neg)
        return emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg

    def reset_parameters(self):
        self.link_mlp.reset_parameters()
        for layer in self.layer_ab_pos:
            layer.reset_parameters()
        for layer in self.layer_ab_neg:
            layer.reset_parameters()

    def predict_edges(self, emb, uid, qid):
        """Predict the sign of edges given embeddings and user id, question id"""
        usr_emb = emb[uid]
        qus_emb = emb[qid]
        return self.link_mlp(usr_emb, qus_emb)

    def compute_label_loss(self, y_score, y_label):
        pos_weight = torch.tensor([(y_label == 0).sum().item() / (y_label == 1).sum().item()] * y_label.shape[0]).to(
            self.device)
        return F.binary_cross_entropy_with_logits(y_score, y_label, pos_weight=pos_weight)

    def compute_contrastive_loss(self, emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg):
        nodes_num, feature_size = emb_g1_pos.shape

        emb_g1_pos = emb_g1_pos.to(self.device)
        emb_g2_pos = emb_g2_pos.to(self.device)
        emb_g1_neg = emb_g1_neg.to(self.device)
        emb_g2_neg = emb_g2_neg.to(self.device)

        norm_emb_g1_pos = F.normalize(emb_g1_pos, p=2, dim=1)
        norm_emb_g2_pos = F.normalize(emb_g2_pos, p=2, dim=1)
        norm_emb_g1_neg = F.normalize(emb_g1_neg, p=2, dim=1)
        norm_emb_g2_neg = F.normalize(emb_g2_neg, p=2, dim=1)

        def inter_contrastive(embs_attr, embs_stru):
            pos = torch.exp(torch.div(
                torch.bmm(embs_attr.view(nodes_num, 1, feature_size), embs_stru.view(nodes_num, feature_size, 1)),
                self.args.tau))

            def generate_neg_score(emb_1, emb_2):
                neg_similarity = torch.mm(emb_1.view(nodes_num, feature_size), emb_2.transpose(0, 1)).fill_diagonal_(0)
                return torch.sum(torch.exp(torch.div(neg_similarity, self.args.tau)), dim=1)

            neg = generate_neg_score(embs_attr, embs_stru)

            return torch.mean(- (torch.log(torch.div(pos, neg))))

        def intra_contrastive(self_embs, embs_attr_pos, embs_attr_neg, embs_stru_pos, embs_stru_neg):
            pos_score_1 = torch.exp(torch.div(
                torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_attr_pos.view(nodes_num, feature_size, 1)),
                self.args.tau))
            pos_score_2 = torch.exp(torch.div(
                torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_stru_pos.view(nodes_num, feature_size, 1)),
                self.args.tau))
            pos = pos_score_1 + pos_score_2

            def generate_neg_score(pos_embs, neg_embs_1, neg_embs_2):
                neg_score_1 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size),
                                        neg_embs_1.view(nodes_num, feature_size, 1))
                neg_score_2 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size),
                                        neg_embs_2.view(nodes_num, feature_size, 1))
                return torch.exp(torch.div(neg_score_1, self.args.tau)) + torch.exp(
                    torch.div(neg_score_2, self.args.tau))

            neg = generate_neg_score(self_embs, embs_attr_neg, embs_stru_neg)
            return torch.mean(- torch.log(torch.div(pos, neg)))

        inter_pos = inter_contrastive(norm_emb_g1_pos, norm_emb_g2_pos)
        inter_neg = inter_contrastive(norm_emb_g1_neg, norm_emb_g2_neg)

        emb = torch.cat((emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg), dim=-1)
        self.embs = self.linear_combine(emb)
        self.norm_embs = F.normalize(self.embs, p=2, dim=1)

        intra = intra_contrastive(self.norm_embs, norm_emb_g1_pos, norm_emb_g1_neg,
                                  norm_emb_g2_pos, norm_emb_g2_neg)
        return (1 - self.args.alpha) * (inter_pos + inter_neg) + self.args.alpha * intra

    # def compute_contrastive_loss(self, emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg):
    #     nodes_num, feature_size = emb_g1_pos.shape
    #     emb_g1_pos, emb_g2_pos = emb_g1_pos.to(self.device), emb_g2_pos.to(self.device)
    #     emb_g1_neg, emb_g2_neg = emb_g1_neg.to(self.device), emb_g2_neg.to(self.device)
    #
    #     def inter_contrastive(emb_1, emb_2):
    #         pos_score = torch.exp(torch.div(torch.bmm(emb_1.view(nodes_num, 1, feature_size),
    #                                         emb_2.view(nodes_num, feature_size, 1)),
    #                               self.args.tau))
    #         neg_sim = torch.mm(emb_1, emb_2.transpose(0, 1)).fill_diagonal_(0)
    #         neg_score = torch.exp(torch.div(neg_sim, self.args.tau)).sum(dim=1)
    #         return torch.mean(- (torch.log(torch.div(pos_score, neg_score))))
    #
    #     def intra_contrastive(self_embs, embs_g1_pos, embs_g1_neg, embs_g2_pos, embs_g2_neg):
    #         pos_score_1 = torch.exp(torch.div(torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_g1_pos.view(nodes_num, feature_size, 1)), self.args.tau))
    #         pos_score_2 = torch.exp(torch.div(torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_g2_pos.view(nodes_num, feature_size, 1)), self.args.tau))
    #         pos = pos_score_1 + pos_score_2
    #
    #         def generate_neg_score(pos_embs, neg_embs_1, neg_embs_2):
    #             neg_score_1 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size), neg_embs_1.view(nodes_num, feature_size, 1))
    #             neg_score_2 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size), neg_embs_2.view(nodes_num, feature_size, 1))
    #             return torch.exp(torch.div(neg_score_1, self.args.tau)) + torch.exp(torch.div(neg_score_2, self.args.tau))
    #
    #         neg = generate_neg_score(self_embs, embs_g1_neg, embs_g2_neg)
    #         return torch.mean(-torch.log(torch.div(pos, neg)))
    #
    #     emb_g1_pos = F.normalize(emb_g1_pos, p=2, dim=1)  # normalize the embeddings (l2)
    #     emb_g2_pos = F.normalize(emb_g2_pos, p=2, dim=1)
    #     emb_g1_neg = F.normalize(emb_g1_neg, p=2, dim=1)
    #     emb_g2_neg = F.normalize(emb_g2_neg, p=2, dim=1)
    #
    #     emb = self.linear_combine(torch.cat([emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg], dim=-1))
    #     emb = F.normalize(emb, p=2, dim=1)
    #     self.norm_embs = emb  # store the normalized embeddings
    #
    #     # inter loss
    #     inter_pos = inter_contrastive(emb_g1_pos, emb_g2_pos)
    #     inter_neg = inter_contrastive(emb_g1_neg, emb_g2_neg)
    #
    #     # intra loss
    #     intra = intra_contrastive(emb, emb_g1_pos, emb_g1_neg, emb_g2_pos, emb_g2_neg)
    #     return (1 - self.args.alpha) * (inter_pos + inter_neg) + self.args.alpha * intra


class SGNNEnc(torch.nn.Module):
    """Learn the embeddings for each node"""

    def __init__(self, usr_num, qus_num, input_dim, args):
        super(SGNNEnc, self).__init__()

        self.args = args
        self.emb_size = args.emb_size  # embedding size
        self.emb_out = None  # output embedding
        self.usr_num, self.qus_num = usr_num, qus_num  # user and question number

        # linear transformation
        self.linear_g1 = torch.nn.Linear(input_dim, self.emb_size, bias=False)  # graph 1
        self.linear_g2 = torch.nn.Linear(input_dim, self.emb_size, bias=False)  # graph 2
        self.linear_g3 = torch.nn.Linear(input_dim, self.emb_size, bias=False)  # graph 3
        self.linear_g4 = torch.nn.Linear(input_dim, self.emb_size, bias=False)  # graph 4

        # signed convolution layers
        self.sconv_p1 = SignedGCN(self.emb_size, self.emb_size, num_layers=args.num_layers)  # perspective 1
        self.sconv_p2 = SignedGCN(self.emb_size, self.emb_size, num_layers=args.num_layers)  # perspective 2

        # transform the concatenated embeddings from 4 graphs
        self.linear_combine = torch.nn.Linear(4 * self.emb_size, self.emb_size, bias=False)
        self.dropout = torch.nn.Dropout(p=args.dropout)

        # link MLP
        self.link_mlp = LinkMLP(args)

        self.reset_parameters()

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
