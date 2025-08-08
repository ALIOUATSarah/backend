from scipy.ndimage import center_of_mass, shift


def center_image(img_array):
    cy, cx = center_of_mass(img_array)
    shift_y = int(14 - cy)
    shift_x = int(14 - cx)
    return shift(img_array, [shift_y, shift_x], mode="constant")
