from typing import Optional, Tuple

import numpy as np
import omni.replicator.core as rep
try:
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
except ImportError:
    from omni.isaac.core.prims.xform_prim import XFormPrim

from internutopia.core.sensor.camera import ICamera


class IsaacsimCamera(ICamera):
    """
    IsaacSim's implementation on `ICamera` class.

    Args:
        name (str): The unique identifier for the camera.
        prim_path (Optional[str]): The primary path associated with the camera.
        rgba (Optional[bool], default=False): Whether to get rgba form the camera or not.
        distance_to_image_plane (Optional[bool], default=False): Whether to get distance_to_image_plane form the camera or not.
        bounding_box_2d_tight (Optional[bool], default=False): Whether to get bounding_box_2d_tight form the camera or not.
        camera_params (Optional[bool], default=False): Whether to get camera_params form the camera or not.
        resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ). Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim (with respect to its parent prim). shape is (3, ). Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim (depends if translation or position is specified). quaternion is scalar-first (w, x, y, z). shape is (4, ). Defaults to None, which means left unchanged.
    """

    def __init__(
        self,
        name: str = 'camera',
        prim_path: Optional[str] = None,
        rgba: Optional[bool] = True,
        distance_to_image_plane: Optional[bool] = False,
        bounding_box_2d_tight: Optional[bool] = False,
        camera_params: Optional[bool] = False,
        resolution: Optional[Tuple[int, int]] = None,
        position: Optional[Tuple[float, float, float]] = None,
        translation: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float, float]] = None,
        focal_length: Optional[float] = None,
        horizontal_aperture: Optional[float] = None,
        vertical_aperture: Optional[float] = None,
        clipping_range: Optional[Tuple[float, float]] = None,
    ):
        self._ensure_replicator_overscan_defaults()
        self.name = name
        self.rgba = rgba
        self.distance_to_image_plane = distance_to_image_plane
        self.bounding_box_2d_tight = bounding_box_2d_tight
        self.camera_params = camera_params
        self.rp = None
        self.rp_annotators = {}
        self.rp = rep.create.render_product(prim_path, resolution)
        self.prim = XFormPrim(prim_path)
        super().__init__()
        if position is not None:
            self.prim.set_world_pose(position, orientation)
        if translation is not None:
            self.prim.set_local_pose(translation, orientation)
        self._apply_camera_intrinsics(
            prim_path=prim_path,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            clipping_range=clipping_range,
        )
        self.init_rp_annotators()

    @staticmethod
    def _ensure_replicator_overscan_defaults() -> None:
        """Initialize Isaac Sim 5.1's unset Replicator overscan settings.

        Replicator 1.12 assumes these global settings exist when reading a GPU
        annotator.  Some headless Isaac Sim 5.1 launches leave them unset,
        which makes its internal overscan resize path subtract ``None`` values.
        Preserve explicitly configured overscan and fill only missing defaults.
        """
        import carb

        settings = carb.settings.get_settings()
        defaults = {
            '/rtx/dataWindowNDC/0': 0.0,
            '/rtx/dataWindowNDC/1': 0.0,
            '/rtx/dataWindowNDC/2': 1.0,
            '/rtx/dataWindowNDC/3': 1.0,
        }
        for key, value in defaults.items():
            if settings.get(key) is None:
                settings.set(key, value)

    @staticmethod
    def _apply_camera_intrinsics(
        *,
        prim_path: str,
        focal_length: Optional[float] = None,
        horizontal_aperture: Optional[float] = None,
        vertical_aperture: Optional[float] = None,
        clipping_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        if (
            focal_length is None
            and horizontal_aperture is None
            and vertical_aperture is None
            and clipping_range is None
        ):
            return

        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        camera = UsdGeom.Camera(prim)
        if focal_length is not None:
            camera.CreateFocalLengthAttr().Set(float(focal_length))
        if horizontal_aperture is not None:
            camera.CreateHorizontalApertureAttr().Set(float(horizontal_aperture))
        if vertical_aperture is not None:
            camera.CreateVerticalApertureAttr().Set(float(vertical_aperture))
        if clipping_range is not None:
            near, far = clipping_range
            camera.CreateClippingRangeAttr().Set(Gf.Vec2f(float(near), float(far)))

    def init_rp_annotators(self):
        if self.rgba:
            self.rp_annotators['rgba'] = rep.AnnotatorRegistry.get_annotator('LdrColor')
            self.rp_annotators['rgba'].attach(self.rp)
        if self.distance_to_image_plane:
            self.rp_annotators['distance_to_image_plane'] = rep.AnnotatorRegistry.get_annotator(
                'distance_to_image_plane'
            )
            self.rp_annotators['distance_to_image_plane'].attach(self.rp)
        if self.bounding_box_2d_tight:
            self.rp_annotators['bounding_box_2d_tight'] = rep.AnnotatorRegistry.get_annotator('bounding_box_2d_tight')
            self.rp_annotators['bounding_box_2d_tight'].attach(self.rp)
        if self.camera_params:
            self.rp_annotators['camera_params'] = rep.AnnotatorRegistry.get_annotator('camera_params')
            self.rp_annotators['camera_params'].attach(self.rp)

    def get_rgba(self) -> np.ndarray:
        """See `ICamera.get_rgba` for documentation."""
        if self.rgba:
            return self.rp_annotators['rgba'].get_data()
        return None

    def get_distance_to_image_plane(self) -> np.ndarray:
        """See `ICamera.get_distance_to_image_plane` for documentation."""
        if self.distance_to_image_plane:
            return self.rp_annotators['distance_to_image_plane'].get_data()
        return None

    def get_bounding_box_2d_tight(self) -> np.ndarray:
        """See `ICamera.get_bounding_box_2d_tight` for documentation."""
        if self.bounding_box_2d_tight:
            return self.rp_annotators['bounding_box_2d_tight'].get_data()
        return None

    def get_camera_params(self) -> np.ndarray:
        """See `ICamera.get_camera_params` for documentation."""
        if self.camera_params:
            return self.rp_annotators['camera_params'].get_data()
        return None

    def cleanup(self) -> None:
        render_product = self.rp
        if render_product is None:
            return
        for annotator in self.rp_annotators.values():
            try:
                annotator.detach(render_product)
            except Exception:
                pass
        self.rp_annotators = {}
        # Isaac Sim 5.1 may stop the application when a render product is
        # destroyed while the next stage is being composed. Each episode uses
        # a distinct camera prim, and the short-lived worker owns final cleanup.
        self.rp = None

    def unwrap(self):
        return self.prim
