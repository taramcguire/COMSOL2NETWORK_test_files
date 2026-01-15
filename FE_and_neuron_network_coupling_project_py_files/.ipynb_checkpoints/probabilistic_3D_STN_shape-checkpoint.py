import siibra
import numpy as np

#class to get random points based on the STN in the Julich Brain CYTOARCHITECTURE
class Julich_brain_cyto_random_point_generator:
    def __init__(self,parcellation, region, space):
        #obtain the brain region required
        self.region = siibra.get_region(parcellation, region)
        
        #get the space required
        self.space = self.region.spaces.get(space)

    
        self.region_map = self.region.get_regional_map(space, maptype="statistical")
        self.NIfTI_region = self.region_map.fetch()
        self.region_data = self.NIfTI_region.get_fdata()
        self.affine = self.NIfTI_region.affine


        # Flatten and mask out zero or invalid voxels
        self.region_data_flat = self.region_data.flatten()
        self.region_data_valid_probs = self.region_data_flat[self.region_data_flat > 0]

        #normalise the probabilities to work with random choice
        self.region_data_valid_probs /= self.region_data_valid_probs.sum()  

        ## Get voxel indices (i,j,k) of valid voxels
        self.region_data_indices = np.argwhere(self.region_data > 0)


    def sample_points_in_voxels(self, number_points):
        coordinates = np.zeros(shape=(number_points, 4))
        for i in range(0,number_points):
            index = np.random.choice(len(self.region_data_valid_probs), p=self.region_data_valid_probs)
            voxel_coord = self.region_data_indices[index]

            # Add random offset in voxel space
            offset = np.random.normal(0, 0.33,3)
            voxel_point = voxel_coord + offset
            
            # Convert to world coordinate
            voxel_point = np.append(voxel_point, 1)
            voxel_point = self.affine @ voxel_point
            coordinates[i] = voxel_point        
        return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
