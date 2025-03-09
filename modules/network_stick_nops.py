import torch.nn as nn
import torch
from torch.nn.functional import normalize
from torch.nn.functional import pad

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, n_participant, alpha):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.n_participant = n_participant
        self.alpha = alpha
        # 1. Instance Head (IH)
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        # 2. Cluster head + Stick-breaking (CH)
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )


    @staticmethod
    def stickbreak(pi, alpha):


        cum_sum = torch.cumsum(pi, dim=1)
        # pad 0 to the beginning of the tensor
        cum_sum = torch.cat((torch.zeros(cum_sum.shape[0], 1, device = torch.device('cuda')), cum_sum), dim=1)
        one_minus_cumulative_sum = 1 - cum_sum
        beta = pi[:, :-1] / one_minus_cumulative_sum[:, :-2]

        # mask out inactive cluster
        mask = (one_minus_cumulative_sum <=0)[:, :-2]
        beta[mask] = 0
        # Replace any out-of-bound beta values with 0 for log_prob calculation
        # replace 1 with 0.9999999 (floating point issue)
        beta[beta > 0.9999999] = 0.9999999
        
        log_prob = torch.distributions.beta.Beta(1, alpha).log_prob(beta.to('cuda'))
        
        # Calculate the mean of log_prob only with non-zero values(sum is not fair)
        log_prob = torch.where(mask, torch.tensor(0.0), log_prob)
        non_zero_log_prob = log_prob[log_prob != 0]
        mean_log_prob = torch.mean(non_zero_log_prob, dim = 0)
        
        return beta, mean_log_prob

    def forward(self, x_i, x_j):

        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)


        # instance head
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)


        # cluster head
        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        # stick-breaking

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move c_i to GPU
        c_i = c_i.to(device)
        c_j = c_j.to(device)


        # Move the result of self.stickbreak(c_i) to GPU
        alpha = torch.tensor(self.alpha).to(device)
        _, s_i = self.stickbreak(c_i, alpha)
        _, s_j = self.stickbreak(c_j, alpha)
        s_i, s_j = s_i.to(device), s_j.to(device)


        return z_i, z_j, c_i, c_j, s_i, s_j 

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    
    def forward_cluster_individual(self, x):
        h = self.resnet(x)
        z =  normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c, z
    
    def forward_latent_representation(self,x):
        h = self.resnet(x)
        return h



