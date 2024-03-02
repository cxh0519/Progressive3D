import os
from dataclasses import dataclass, field

import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import open3d as o3d
import numpy as np
import math, random, copy
import torch.nn.functional as F

def camera_info_process(batch):
    fovy_list = []
    center_list = []
    position_list = []
    up_list = []

    for b in range(batch['fovy'].shape[0]):
        fovy_list.append(batch['fovy'][b].item())
        center_list.append([batch['center'][b][i].item() for i in range(3)])
        position_list.append([batch['camera_positions'][b][i].item() for i in range(3)])
        up_list.append([batch['up'][b][i].item() for i in range(3)])

    return fovy_list, center_list, position_list, up_list

@threestudio.register("mvdream-progressive3d-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

        box_info: list = field(default_factory=list)
        move_camera: bool = True

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        prompt_processor_ori = copy.deepcopy(self.cfg.prompt_processor)
        prompt_processor_ori["prompt"] = self.cfg.prompt_processor["prompt"].split("|")[0]
        
        self.prompt_processor_ori = threestudio.find(self.cfg.prompt_processor_type)(
            prompt_processor_ori
        )
        self.prompt_utils_ori = self.prompt_processor_ori()

        prompt_processor_edit = copy.deepcopy(self.cfg.prompt_processor)
        prompt_processor_edit["prompt"] = self.cfg.prompt_processor["prompt"].split("|")[1]
        self.prompt_processor_edit = threestudio.find(self.cfg.prompt_processor_type)(
            prompt_processor_edit
        )
        self.prompt_utils_edit = self.prompt_processor_edit()
        self.opacity_threshold = 0.8
        self.move_distance = None

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.geometry_freeze = copy.deepcopy(self.geometry)
        self.material_freeze = copy.deepcopy(self.material)
        self.background_freeze = copy.deepcopy(self.background)

        self.renderer_freeze = copy.deepcopy(self.renderer)
        self.renderer_freeze.set_geometry(self.geometry_freeze)
        self.renderer_freeze.set_material(self.material_freeze)
        self.renderer_freeze.set_background(self.background_freeze)

        self.geometry_freeze.requires_grad_(False)
        self.material_freeze.requires_grad_(False)
        self.background_freeze.requires_grad_(False)

        self.box_render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultLit'

        self.bounding_render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
        bmat = o3d.visualization.rendering.MaterialRecord()
        bmat.shader = "unlitLine"
        bmat.line_width = 3

        if self.cfg.move_camera:
            self.move_distance = [self.cfg.box_info[0][i] / 2.0 for i in [3, 4, 5]]

        for idx, box_info in enumerate(self.cfg.box_info):
            box = o3d.geometry.TriangleMesh.create_box(box_info[0], box_info[1], box_info[2])
            box.compute_vertex_normals()

            box.translate([-box_info[0]/2., -box_info[1]/2., -box_info[2]/2.])
            box.translate([box_info[3], box_info[4], box_info[5]])

            bounding = o3d.geometry.OrientedBoundingBox()
            bounding.center=[0.0, 0.0, 0.0]
            bounding.extent=[box_info[0], box_info[1], box_info[2]]
            bounding.translate([box_info[3], box_info[4], box_info[5]])
            bounding.color = (0, 176/255., 240/255.)

            if self.cfg.move_camera:
                box.translate([-i for i in self.move_distance])
                bounding.translate([-i for i in self.move_distance])

            self.box_render.scene.add_geometry("box"+str(idx), box, mat)
            self.bounding_render.scene.add_geometry("bounding"+str(idx), bounding, bmat)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.material.random_num = self.material_freeze.random_num = random.random()
        render_out = self.renderer(**batch)
        with torch.no_grad():
            render_out_freeze = self.renderer_freeze(**batch) 
        return {
            **render_out, 
            "comp_rgb_freeze": render_out_freeze["comp_rgb"],
            "opacity_freeze": render_out_freeze["opacity"],
            "depth_freeze": render_out_freeze["depth"],
        }

    def training_step(self, batch, batch_idx):
        if self.cfg.move_camera:
            for idx, i in enumerate(self.move_distance):
                batch["rays_o"][..., idx] += i

        out = self(batch)

        fovy, center, position, up = camera_info_process(batch)

        dimg_list = []
        for b in range(batch['fovy'].shape[0]):
            self.box_render.scene.camera.set_projection(fovy[b], 1.0, 0.1, 1000, o3d.visualization.rendering.Camera.FovType.Vertical)
            self.box_render.scene.camera.look_at(center[b], position[b], up[b])

            dimg = torch.tensor(np.asarray(self.box_render.render_to_depth_image(z_in_view_space=True))).to(out["comp_rgb"].device)
            dimg = F.interpolate(
                dimg[None, None, ...], (batch['height'], batch['width']), mode="bilinear", align_corners=False
            ).view(batch['height'], batch['width'], 1)
            dimg_list.append(dimg)
        dimg = torch.stack(dimg_list, dim=0)

        mask_opacity = torch.where(out['opacity_freeze'] > self.opacity_threshold, 1.0, 0.0)

        depth = out['depth_freeze'] * mask_opacity
        depth[depth==0] = float('inf')
        mask = torch.where(dimg < depth, 1., 0.)

        guidance_out = self.guidance(out["comp_rgb"], [self.prompt_utils_ori, self.prompt_utils_edit], mask=mask, **batch)

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        ## Loss
        loss_consistency = F.mse_loss(out["comp_rgb"]*(1-mask)*mask_opacity, out["comp_rgb_freeze"]*(1-mask)*mask_opacity, reduction="sum")
        loss_opacity = F.mse_loss(out["opacity"]*(1-mask)*(1-mask_opacity), torch.zeros_like(out['opacity']).to(out["opacity"].device), reduction="sum")
        loss += (loss_opacity + loss_consistency) * self.C(self.cfg.loss.lambda_consistency)

        loss_init = F.mse_loss(out["opacity"]*mask, torch.ones_like(out['opacity']).to(out["opacity"].device)*mask, reduction="sum")
        loss += loss_init * self.C(self.cfg.loss.lambda_init)

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > self.opacity_threshold].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.cfg.move_camera:
            for idx, i in enumerate(self.move_distance):
                batch["rays_o"][..., idx] += i
        out = self(batch)
        
        fovy, center, position, up = camera_info_process(batch)

        self.box_render.scene.camera.set_projection(fovy[0], 1.0, 0.1, 1000, o3d.visualization.rendering.Camera.FovType.Vertical)
        self.box_render.scene.camera.look_at(center[0], position[0], up[0])

        self.bounding_render.scene.camera.set_projection(fovy[0], 1.0, 0.1, 1000, o3d.visualization.rendering.Camera.FovType.Vertical)
        self.bounding_render.scene.camera.look_at(center[0], position[0], up[0])

        dimg = np.asarray(self.box_render.render_to_depth_image(z_in_view_space=True))
        
        out = self(batch)
        depth = out['depth_freeze'][0] * torch.where(out['opacity_freeze'][0] > self.opacity_threshold, 1.0, 0.0)
        depth[depth==0] = float('inf')

        dimg = torch.tensor(dimg)[..., None].to(depth.device)
        mask = torch.where(dimg < depth, 1., 0.)[..., 0]

        bounding_rendering = torch.tensor(np.asarray(self.bounding_render.render_to_image()), device="cuda").float() / 255.
        bounding_mask = torch.ones(bounding_rendering.shape[:1], device="cuda")[..., None]
        for i in range(bounding_rendering.shape[-1]):
            bounding_mask = bounding_mask * torch.where(bounding_rendering[..., i] > 0.9, 1.0, 0.0)[..., None]
        bounding_rendering = bounding_rendering * (1-bounding_mask) + out["comp_rgb_freeze"][0] * bounding_mask

        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "grayscale",
                    "img": mask,
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": bounding_rendering,
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        if self.cfg.move_camera:
            for idx, i in enumerate(self.move_distance):
                batch["rays_o"][..., idx] += i
        out = self(batch)
        
        fovy, center, position, up = camera_info_process(batch)

        self.box_render.scene.camera.set_projection(fovy[0], 1.0, 0.1, 1000, o3d.visualization.rendering.Camera.FovType.Vertical)
        self.box_render.scene.camera.look_at(center[0], position[0], up[0])

        self.bounding_render.scene.camera.set_projection(fovy[0], 1.0, 0.1, 1000, o3d.visualization.rendering.Camera.FovType.Vertical)
        self.bounding_render.scene.camera.look_at(center[0], position[0], up[0])

        dimg = np.asarray(self.box_render.render_to_depth_image(z_in_view_space=True))
        
        out = self(batch)
        depth = out['depth_freeze'][0] * torch.where(out['opacity_freeze'][0] > self.opacity_threshold, 1.0, 0.0)
        depth[depth==0] = float('inf')

        dimg = torch.tensor(dimg)[..., None].to(depth.device)
        mask = torch.where(dimg < depth, 1., 0.)[..., 0]

        bounding_rendering = torch.tensor(np.asarray(self.bounding_render.render_to_image()), device="cuda").float() / 255.
        bounding_mask = torch.ones(bounding_rendering.shape[:1], device="cuda")[..., None]
        for i in range(bounding_rendering.shape[-1]):
            bounding_mask = bounding_mask * torch.where(bounding_rendering[..., i] > 0.9, 1.0, 0.0)[..., None]
        bounding_rendering = bounding_rendering * (1-bounding_mask) + out["comp_rgb_freeze"][0] * bounding_mask
        
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "grayscale",
                    "img": mask,
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": bounding_rendering,
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
    
    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {k: v for k, v in state_dict.items() if not "_freeze" in k}
        super().load_state_dict(new_state_dict, strict=strict)
