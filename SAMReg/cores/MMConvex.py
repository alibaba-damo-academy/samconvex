from syslog import LOG_SYSLOG
import torch
import torch.nn as nn
import torch.nn.functional as F

from SAMReg.cores.functionals import spatial_transformer, correlation, correlation_split, compose, neg_Jdet_loss, JacboianDet


class MMConvex(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed
        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        # self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")

    def forward(self, source, target, pre_align=None, pre_align_flow=None):
        if pre_align is not None:
            # ## Warp source with affine
            initial_phi = (
                F.affine_grid(pre_align, size=source.shape, align_corners=True)
                .permute(0, 4, 1, 2, 3)
                .flip(1)
            )
            ### Warp source with coarse/flow
            # initial_phi = (
            #     F.interpolate(pre_align, size=source.shape[2:], mode="trilinear", align_corners=True)
            # )
            warped = spatial_transformer(source, initial_phi, mode='bilinear', padding_mode="background")
        else:
            warped = source


        with torch.no_grad():
            target_2 = self.down_avg(target)
            warped_2 = self.down_avg(warped)

            target_4 = self.down_avg(target_2)
            warped_4 = self.down_avg(warped_2)

            # ------------------level 1/4------------------

            disp_hr_4 = self._Mconvex(warped_4, target_4, kernel_size=2)

            disp_hr_4 = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(disp_hr_4, 3, padding=1, stride=1), 3, padding=1, stride=1
                ),
                3,
                padding=1,
                stride=1
            )
            

            # # ------------------from level 1/4 to level 1/2------------------

            # up_disp_hr_4 = self.up_tri(disp_hr_4) 
            up_disp_hr_4 = F.interpolate(disp_hr_4, size=warped_2.shape[2:], mode="trilinear", align_corners=True)

            grid = (
                F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0).cuda(),
                    warped_2.shape,
                    align_corners=True,
                ).permute(0, 4, 1, 2, 3).flip(1)
            )
            warped_2 = spatial_transformer(
                warped_2, 
                grid + up_disp_hr_4, 
                mode="bilinear", padding_mode="background"
            )

            # # ------------------level 1/2------------------

            disp_hr_2 = self._Mconvex(warped_2, target_2, kernel_size=3)

            disp_hr_2 = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(disp_hr_2, 3, padding=1, stride=1), 3, padding=1, stride=1
                ),
                3,
                padding=1,
                stride=1
            )

            disp_hr_2 = compose(up_disp_hr_4, disp_hr_2)


            # # # ------------------from level 1/2 to level 1------------------

            # up_disp_hr_2 = self.up_tri(disp_hr_2) 
            up_disp_hr_2 = F.interpolate(disp_hr_2, size=warped.shape[2:], mode="trilinear", align_corners=True)

            grid = (
                F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0).cuda(),
                    warped.shape,
                    align_corners=True,
                ).permute(0, 4, 1, 2, 3).flip(1)
            )
            warped = spatial_transformer(
                warped, 
                grid + up_disp_hr_2, 
                mode="bilinear", padding_mode="background"
            )

            # ------------------level 1------------------

            disp_hr = self._MconvexFine(warped, target, kernel_size=3)

            disp_hr = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(disp_hr, 3, padding=1, stride=1), 3, padding=1, stride=1
                ),
                3,
                padding=1,
                stride=1
            )

            disp_hr = compose(up_disp_hr_2, disp_hr)


        disp_smooth = F.avg_pool3d(
            F.avg_pool3d(
                F.avg_pool3d(disp_hr, 3, padding=1, stride=1), 3, padding=1, stride=1
            ),
            3,
            padding=1,
            stride=1,
        )

        if pre_align is not None:
            deform_phi = grid + disp_smooth
            phi = grid + compose((initial_phi - grid), disp_smooth)
        else:
            phi = grid + disp_smooth
            deform_phi = phi

        warped = spatial_transformer(
            source, phi, mode="bilinear", padding_mode="background"
        )
        return warped, phi, deform_phi



    def _adam_convex(self, feat_source_coarse, feat_target_coarse, kernel_size):
        cost_volume = -correlation_split(
            feat_source_coarse.half(), feat_target_coarse.half(), kernel_size, 1, kernel="ssd"
        )

        disp_soft = self._coupled_convex(cost_volume, kernel_size).float()

        return disp_soft

    # solve two coupled convex optimisation problems for efficient global regularisation
    def _coupled_convex(self, cost, kernel_size):
        b, _, h, w, d = cost.shape
        assert b == 1, "MMConvex supports registration of one pair of images only!"

        cost = cost[0]
        cost_argmin = torch.argmin(cost, 0)

        disp_mesh_t = (
            F.affine_grid(
                kernel_size * torch.eye(3, 4).cuda().half().unsqueeze(0),
                (1, 1, kernel_size * 2 + 1, kernel_size * 2 + 1, kernel_size * 2 + 1),
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .flip(1)
            .reshape(3, -1, 1)
        )
        # Normalize to [-1,1]
        scale = torch.tensor([h - 1, w - 1, d - 1]).view(3, 1, 1).cuda() / 2.0
        disp_mesh_t = disp_mesh_t / scale

        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3, -1)[:, cost_argmin.view(-1)].reshape(1, 3, h, w, d),
            3,
            padding=1,
            stride=1,
        )
        

        coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
        for j in range(6):
            ssd_coupled_argmin = torch.zeros_like(cost_argmin)
            with torch.no_grad():
                for i in range(h):
                    coupled = cost[:, i, :, :] + coeffs[j] * (
                        disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)
                    ).pow(2).sum(0).view(-1, w, d)
                    ssd_coupled_argmin[i] = torch.argmin(coupled, 0)

            disp_soft = F.avg_pool3d(
                disp_mesh_t.view(3, -1)[:, ssd_coupled_argmin.view(-1)].reshape(
                    1, 3, h, w, d
                ),
                3,
                padding=1,
                stride=1,
            )
                #################
            disp_soft = F.avg_pool3d(
                F.avg_pool3d(
                    disp_soft, 3, padding=1, stride=1
                ),
                3,
                padding=1,
                stride=1,
            )
            ################

        return disp_soft


    def _instance_optimization(self, param, feat_source, feat_target):

        param = nn.parameter.Parameter(param)  # B*3*H*W*D

        optimizer = torch.optim.Adam([param], lr=0.05, eps=1e-4)
        grid0 = (
            F.affine_grid(
                torch.eye(3, 4).unsqueeze(0).to(param.device),
                param.shape,
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .flip(1)
        )

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = 1000  # with tps: .5, without:0.7
        for iter in range(50):  # 80
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(param, 3, stride=1, padding=1),
                    3,
                    stride=1,
                    padding=1,
                ),
                3,
                stride=1,
                padding=1,
            )
            reg_loss = (
                lambda_weight
                * (
                    (disp_sample[0, :, :, 1:, :] - disp_sample[0, :, :, :-1, :]) ** 2
                ).mean()
                + lambda_weight
                * (
                    (disp_sample[0, :, 1:, :, :] - disp_sample[0, :, :-1, :, :]) ** 2
                ).mean()
                + lambda_weight
                * (
                    (disp_sample[0, :, :, :, 1:] - disp_sample[0, :, :, :, :-1]) ** 2
                ).mean()
            )

            phi = grid0 + disp_sample

            feat_dim = feat_source[0].shape[1]
            feat_warped = spatial_transformer(
                torch.cat(feat_source, 1),
                phi,
                mode="bilinear",
                padding_mode="border",
            )

            sampled_cost = -F.cosine_similarity(
                F.normalize(feat_warped[:, :feat_dim].half(), 1),
                feat_target[0].half(),
                1,
            )# - F.cosine_similarity(
            #     F.normalize(feat_warped[:, feat_dim:].half(), 1),
            #     feat_target[1].half(),
            #     1,
            # )
            loss = sampled_cost.mean()
            (loss + reg_loss).backward()
            optimizer.step()


        return param.detach()
    

    def _Mconvex(self, source, target, kernel_size):
        with torch.no_grad():
            # Get SAM feature
            target_fine, target_coarse = self.embed.extract_feat(target)
            source_fine, source_coarse = self.embed.extract_feat(source)
            
            source_coarse = F.normalize(
                F.interpolate(
                    source_coarse,
                    size=source_fine.shape[2:],
                    mode="trilinear",
                    align_corners=True
                ), dim=1
            )
            target_coarse = F.normalize(
                F.interpolate(
                    target_coarse,
                    size=source_fine.shape[2:],
                    mode="trilinear",
                    align_corners=True
                ), dim=1
            )

            disp = self._adam_convex(
                torch.cat([source_fine, source_coarse], dim=1),
                torch.cat([target_fine, target_coarse], dim=1),
                kernel_size,
            )


            disp_hr = F.interpolate(
                disp, size=target.shape[2:], mode="trilinear", align_corners=True
            )
        return disp_hr
    

    def _MconvexFine(self, source, target, kernel_size):
        with torch.no_grad():
            # Get SAM feature
            target_fine, target_coarse = self.embed.extract_feat(target)
            source_fine, source_coarse = self.embed.extract_feat(source)
        
            disp = self._adam_convex(
                source_fine,
                target_fine,
                kernel_size,
            )

            disp_hr = F.interpolate(
                disp, size=target.shape[2:], mode="trilinear", align_corners=True
            )
        return disp_hr
    
    
    def instanceOptimization(self, source, target, pre_align=None):
        if pre_align is not None:
            pre_align = F.interpolate(pre_align, size=source.shape[2:], mode='trilinear', align_corners=True)
            warped  = spatial_transformer(source, pre_align, mode='bilinear', padding_mode="background")
        else:
            warped = source
        # warped = source

        with torch.no_grad():
            # Get SAM feature
            
            target_fine = self.embed.extract_feat(target)[0]
            source_fine = self.embed.extract_feat(warped)[0]

            grid = (
                F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0).cuda(),
                    source.shape,
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .flip(1)
            )

        disp_init = torch.zeros((1, 3) + source_fine.shape[2:]).cuda()

        fitted_disp = self._instance_optimization(
            disp_init,
            [source_fine.float()],
            [target_fine.float()],
        )

        disp_hr = F.interpolate(
            fitted_disp, size=source.shape[2:], mode="trilinear", align_corners=True
        )

        disp_smooth = F.avg_pool3d(
            F.avg_pool3d(
                F.avg_pool3d(disp_hr, 3, padding=1, stride=1), 3, padding=1, stride=1
            ),
            3,
            padding=1,
            stride=1,
        )

        if pre_align is not None:
            phi = grid + compose(pre_align - grid, disp_smooth)
        else:
            phi = grid + disp_smooth 
        

        warped = spatial_transformer(
            source, phi, mode="bilinear", padding_mode="background"
        )
        return warped, phi