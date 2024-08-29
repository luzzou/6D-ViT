import torch
import torch.nn as nn
from lib.rgbformer import RGBformer
from lib.pcformer_msmh import PCformer
import torch.nn.functional as F

class DeformNet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024):
        super(DeformNet, self).__init__()
        self.n_cat = n_cat

        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        # PointNet-like architecture
        self.instance_geometry = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )

        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.category_local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.category_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )

        self.rgbformer = RGBformer(
            dims=(32, 64, 160, 256),  # dimensions of each stage
            heads=(1, 2, 5, 8),  # heads of each stage
            ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
            reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
            num_layers=2,  # num layers of each stage
            decoder_dim=256  # encoder dimension
        )

        self.pcformer = PCformer(
            dims=(32, 64, 160, 256),  # dimensions of each stage
            heads=(1, 2, 5, 8),  # heads of each stage
            ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
            reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
            num_layers=2,  # num layers of each stage
            decoder_dim=64  # decoder dimension
        )


        # Initialize weights to be small so initial deformations aren't so big
        self.deformation[4].weight.data.normal_(0, 0.0001)

    def forward(self, points, img, choose, cat_id, prior):
        """
        Args:
            points: bs x n_pts x 3
            img: bs x 3 x H x W
            choose: bs x n_pts
            cat_id: bs
            prior: bs x nv x 3

        Returns:
            assign_mat: bs x n_pts x nv
            inst_shape: bs x nv x 3
            deltas: bs x nv x 3
            log_assign: bs x n_pts x nv, for numerical stability

        """
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]  # 1024
        # instance-specific features
        points = points.permute(0, 2, 1)  # bs,3,1024
        points = self.instance_geometry(points)  # 3-64-64-64   this is the pointnet-like architecture
        points = self.pcformer(points)  #bs,64,1024
        out_img = self.rgbformer(img)  # bs,32,256,256
        di = out_img.size()[1]  # 32
        emb = out_img.view(bs, di, -1)  # bs,32,256*256
        choose = choose.unsqueeze(1).repeat(1, di, 1)  # bs,32,1024
        emb = torch.gather(emb, 2, choose).contiguous()  # bs,32,1024
        emb = self.instance_color(emb)  # 一层conv1d, bs,64,1024

        inst_local = torch.cat((points, emb), dim=1)  # bs x 128 x n_pts
        inst_global = self.instance_global(inst_local)  # bs x 1024 x 1
        # category-specific features
        cat_prior = prior.permute(0, 2, 1)  # bs,3,1024
        cat_local = self.category_local(cat_prior)  # bs x 64 x n_pts, 对shape prior提取特征
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1，对shape prior提取特征(PointNet结构的特征提取)
        # assignment matrix, aka. correspondence matrix

        assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)),
                                dim=1)  # bs x 2176 x n_pts 融合rgb,point_cloud和shape prior的特征
        assign_mat = self.assignment(assign_feat)  # conv1d, 2176-512-256-(6*1024): n_cat*n_prior    bs,6144,1024
        assign_mat = assign_mat.view(-1, nv,
                                     n_pts).contiguous()  # bs, nc*nv, n_pts -> bs*nc, nv, n_pts   bs*6,1024,1024
        index = cat_id + torch.arange(bs,
                                      dtype=torch.long).cuda() * self.n_cat  ##TODO:?   0-186,间隔6，bs个值，再加上batch中的cat_id
        assign_mat = torch.index_select(assign_mat, 0, index)  # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()  # bs x n_pts x nv
        # deformation field
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)),
                                dim=1)  # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)  # conv1d: 2112-512-256-6*3
        deltas = deltas.view(-1, 3, nv).contiguous()  # bs, nc*3, nv -> bs*nc, 3, nv   bs*6,3,1024
        deltas = torch.index_select(deltas, 0, index)  # bs x 3 x nv
        deltas = deltas.permute(0, 2,
                                1).contiguous()  # bs x nv x 3     bs,1024,3   nv： prior   n_pts: object model point cloud

        return assign_mat, deltas  # bs x n_pts x nv,     bs x nv x 3