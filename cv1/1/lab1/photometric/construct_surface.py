import numpy as np

def construct_surface(p, q, path_type='column'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """
        # Integrate left-most column.
        height_map[1:,0] = np.cumsum(q[1:,0])
        # Integrate rows.
        height_map[:,1:] = np.cumsum(p[:,1:], axis=1)

    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
        # Integrate top row.
        height_map[0,1:] = np.cumsum(p[0,1:])
        # Integrate columns.
        height_map[1:,:] = np.cumsum(q[1:,:], axis=0)

    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """
        # Compute column- and row-major height maps.
        height_map_cm = construct_surface(p, q, path_type='column')
        height_map_rm = construct_surface(p, q, path_type='row')

        # Add and average.
        height_map = (height_map_cm + height_map_rm) / 2

    return height_map

