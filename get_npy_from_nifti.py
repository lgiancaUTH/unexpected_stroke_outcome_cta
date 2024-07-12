import nilearn.image as ni_img
import numpy as np
import config

whole_brain = ni_img.load(config.BRAIN_NIFTI_PATH)
brain_array = whole_brain.get_fdata().astype(int)
h = brain_array.shape[0]
left_brain = brain_array[int(h/2):h, :, :]
right_brain = brain_array[0:int(h/2), :, :]
right_brain = np.flip(right_btain, axis=0)
left = np.clip(left_brain, 0, 100)
right = np.clip(right_brain, 0, 100)
brain_dict = {}
brain_dict['leftBrain'] = left.astype(int)
brain_dict['rightBrain'] = right.astype(int)
np.save(config.IMG_PATH, brain_dict)