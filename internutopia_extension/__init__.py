def import_extensions():
    try:
        import carb  # noqa: F401
    except ModuleNotFoundError:
        # The pip Isaac Sim distribution registers Kit modules when isaacsim is imported.
        import isaacsim  # noqa: F401

    import internutopia_extension.controllers
    import internutopia_extension.interactions
    import internutopia_extension.metrics
    import internutopia_extension.objects
    import internutopia_extension.robots
    import internutopia_extension.sensors
    import internutopia_extension.tasks
