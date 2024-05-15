from pxr import UsdGeom
from omni.isaac.core.utils.stage import get_current_stage
from torchvision.utils import save_image, make_grid
from torchvision.io import write_video, read_image
import os
import torch


def CameraTaskDecorator(cls):
    class CameraTask(cls):
        def __init__(self, name, sim_config, env, offset=None) -> None:
            sim_config.config['train']['params']['config']['horizon_length'] = 100
            sim_config.config['train']['params']['config']['minibatch_size'] = \
                sim_config.config['train']['params']['config']['horizon_length'] * \
                sim_config.config['train']['params']['config']['num_actors']

            super().__init__(name, sim_config, env, offset)
            self.progress = 0
            self.camera_type = sim_config.config['task']['env'].get("cameraType", 'rgb')
            self.camera_width = 1000
            self.camera_height = 1000
            self.camera_channels = 3
            self._export_images = True

            if 'images' not in os.listdir('.'):
                os.mkdir('images')
            if 'videos' not in os.listdir('.'):
                os.mkdir('videos')

        def add_camera(self) -> None:
            stage = get_current_stage()

            attached_prim = str(stage.GetPrimAtPath('/World/envs/env_0').GetChildren()[0].GetName())
            camera_path = f"/World/envs/env_0/{attached_prim}/Camera"
            camera_xform = stage.DefinePrim(f'{camera_path}_Xform', 'Xform')

            # set up transforms for parent and camera prims
            UsdGeom.Xformable(camera_xform).AddTranslateOp()
            # UsdGeom.Xformable(camera_xform).AddRotateXYZOp()
            camera_xform.GetAttribute('xformOp:translate').Set((-7., 0.0, 0.0))
            # camera_xform.GetAttribute('xformOp:rotateXYZ').Set(rotation)
            camera = stage.DefinePrim(f'{camera_path}_Xform/Camera', 'Camera')

            UsdGeom.Xformable(camera).AddRotateXYZOp()
            UsdGeom.Xformable(camera).AddTranslateOp()
            camera.GetAttribute("xformOp:translate").Set((0., 0., 0.))
            camera.GetAttribute("xformOp:rotateXYZ").Set((90., 0, -90.))

            # set camera properties
            camera.GetAttribute('focalLength').Set(24)
            camera.GetAttribute('focusDistance').Set(400)

            # hide other environments in the background
            camera.GetAttribute("clippingRange").Set((5., 10.))

        def set_up_scene(self, scene) -> None:
            super().set_up_scene(scene)
            self.add_camera()

            # start replicator to capture image data
            self.rep.orchestrator._orchestrator._is_started = True

            # set up cameras
            self.render_products = []
            env_pos = self._env_pos.cpu()

            attached_prim = str(scene.stage.GetPrimAtPath('/World/envs/env_0').GetChildren()[0].GetName())
            camera_paths = [f"/World/envs/env_{i}/{attached_prim}/Camera_Xform/Camera" for i in range(self._num_envs)]
            for i in range(self._num_envs):
                render_product = self.rep.create.render_product(camera_paths[i],
                                                                resolution=(self.camera_width, self.camera_height))
                self.render_products.append(render_product)

            # initialize pytorch writer for vectorized collection
            self.pytorch_listener = self.PytorchListener()
            self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
            self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
            self.pytorch_writer.attach(self.render_products)
            return

        def get_observations(self) -> dict:
            observations = super().get_observations()

            # retrieve RGB data from all render products
            img = self.pytorch_listener.get_rgb_data()
            if img is not None:
                if self._export_images:
                    for i in range(self.num_envs):
                        if f'env_{i}' not in os.listdir('images'):
                            os.mkdir(f'images/env_{i}')
                        save_image(img[i, ...] / 255, f'images/env_{i}/img_{self.progress_buf[i]}.png')
            else:
                print("Image tensor is NONE!")

            return observations

        def is_done(self) -> None:
            super().is_done()
            if (self.reset_buf == 1).any():
                self.progress += 1
                done_envs = torch.nonzero(self.reset_buf).flatten()
                for i in done_envs.tolist():
                    # list all images in the directory using os.listdir
                    image_files = [img for img in os.listdir(f'images/env_{i}') if img.endswith(".png")]
                    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                    # load all images
                    images = torch.stack([read_image(f'images/env_{i}/{img}').permute(1, 2, 0) for img in image_files])
                    write_video(f'videos/video_{self.progress}.mp4', images, fps=30)
                    # remove all images
                    for img in image_files:
                        os.remove(f'images/env_{i}/{img}')

    return CameraTask
