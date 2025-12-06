def get_geometry_mask(geometry, grid):

    geometry_type = geometry.type
    if geometry_type == "circle":
        mask = (grid[..., 0]-geometry.center.x)**2 + \
            (grid[..., 1] - geometry.center.y)**2 <= geometry.radius**2
    elif geometry_type == "rect":
        x1, y1 = geometry.origin.x, geometry.origin.y
        x2, y2 = x1+geometry.size.width, y1+geometry.size.height

        cellx = grid[..., 0]
        celly = grid[..., 1]
        mask = (cellx >= x1) & (cellx <= x2) & (celly >= y1) & (celly <= y2)

    return mask
