from pathlib import Path


def bind_preview_surface_texture(
    prim_path: str,
    *,
    texture_path: str,
    texture_scale: tuple[float, float] = (1.0, 1.0),
    texture_rotation_degrees: float = 0.0,
    surface_scale: tuple[float, float, float] | None = None,
    surface_position: tuple[float, float, float] | None = None,
) -> None:
    """Bind a repeatable UsdPreviewSurface texture to an existing object prim."""

    resolved_texture_path = Path(texture_path).expanduser().resolve()
    if not resolved_texture_path.is_file():
        raise FileNotFoundError(f'Table texture does not exist: {resolved_texture_path}')

    try:
        from isaacsim.core.utils.prims import get_prim_at_path
    except ImportError:
        from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    prim = get_prim_at_path(prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f'Cannot bind a table texture to invalid prim {prim_path!r}.')
    stage = prim.GetStage()
    # VisualCuboid can expose either the root prim or a child Cube/Mesh as its
    # renderable Gprim depending on the Isaac Sim API version.  Author UVs and
    # bind the material to every renderable prim so the result is version-safe.
    bind_prims = [candidate for candidate in Usd.PrimRange(prim) if candidate.IsA(UsdGeom.Gprim)]
    if not bind_prims:
        bind_prims = [prim]

    uv_cycle = [
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(0.0, 1.0),
    ]
    for bind_prim in bind_prims:
        uv_count = 24
        if bind_prim.IsA(UsdGeom.Mesh):
            face_counts = UsdGeom.Mesh(bind_prim).GetFaceVertexCountsAttr().Get() or []
            uv_count = max(sum(int(value) for value in face_counts), 4)
        uv_values = Vt.Vec2fArray([uv_cycle[index % len(uv_cycle)] for index in range(uv_count)])
        UsdGeom.PrimvarsAPI(bind_prim).CreatePrimvar(
            'st',
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        ).Set(uv_values)

    safe_prim_name = prim_path.strip('/').replace('/', '_') or 'surface'
    looks_path = Sdf.Path('/World/Looks')
    UsdGeom.Scope.Define(stage, looks_path)
    material_path = f'/World/Looks/domain_randomized_table_material_{safe_prim_name}'
    material = UsdShade.Material.Define(stage, material_path)

    shader = UsdShade.Shader.Define(stage, f'{material_path}/PreviewSurface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.62)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)

    primvar = UsdShade.Shader.Define(stage, f'{material_path}/PrimvarReader')
    primvar.CreateIdAttr('UsdPrimvarReader_float2')
    primvar.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    primvar_output = primvar.CreateOutput('result', Sdf.ValueTypeNames.Float2)

    transform = UsdShade.Shader.Define(stage, f'{material_path}/Transform2d')
    transform.CreateIdAttr('UsdTransform2d')
    transform.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(primvar_output)
    transform.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(float(texture_scale[0]), float(texture_scale[1]))
    )
    transform.CreateInput('rotation', Sdf.ValueTypeNames.Float).Set(float(texture_rotation_degrees))
    transform_output = transform.CreateOutput('result', Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, f'{material_path}/AlbedoTexture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(resolved_texture_path))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
    texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(transform_output)
    texture_output = texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)

    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture_output)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    # Isaac's preview shader resolves the UV primvar through this material
    # interface; setting it explicitly avoids falling back to displayColor.
    material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token).Set('st')
    primvar.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(
        material.GetInput('frame:stPrimvarName')
    )
    if surface_scale is not None and surface_position is not None:
        # VisualCuboid's renderable prim differs between Isaac Sim releases.
        # Add a thin, non-colliding UV mesh in the local frame as a stable
        # visible overlay; it inherits the cube transform and does not alter
        # the physics shape used by the task.
        dimensions = [abs(float(value)) for value in surface_scale]
        thin_axis = min(range(3), key=dimensions.__getitem__)
        plane_axes = [axis for axis in range(3) if axis != thin_axis]
        overlay_path = f'{prim_path}_domain_randomization_texture_surface'
        overlay = UsdGeom.Mesh.Define(stage, overlay_path)
        points = []
        for first, second in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            point = [0.0, 0.0, 0.0]
            point[thin_axis] = dimensions[thin_axis] * 0.5 + 0.0001
            point[plane_axes[0]] = first * dimensions[plane_axes[0]]
            point[plane_axes[1]] = second * dimensions[plane_axes[1]]
            points.append(Gf.Vec3f(*point))
        overlay.CreatePointsAttr(points)
        overlay.CreateFaceVertexCountsAttr([4])
        overlay.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        overlay.CreateExtentAttr([Gf.Vec3f(*points[0]), Gf.Vec3f(*points[2])])
        overlay.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in surface_position]))
        UsdGeom.PrimvarsAPI(overlay.GetPrim()).CreatePrimvar(
            'st',
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        ).Set(Vt.Vec2fArray(uv_cycle))
        bind_prims.append(overlay.GetPrim())
    for bind_prim in bind_prims:
        UsdShade.MaterialBindingAPI.Apply(bind_prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
