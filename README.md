# Image-Panaroma-using-ORB-feature-matching
This project aims to stitch multiple images into a single panoramic photo. The given images may be non-overlapping or multiply overlapped, resulting in an irregularly shaped output image. The project allows for flexibility in the method used to stitch the photos together.
# Assumptions
* The code will need to stitch together four or more images, with the number of images not known in advance.
* If an image is to be part of the panorama, it will overlap at least one other image by at least 20%.
* Images that do not overlap with any other image can be ignored.
* Images can overlap with multiple images.
* The images can be stitched together in any orientation, not just horizontally.
* Only one overall image is expected as the output.
* Basic 2D planar transformations are sufficient for this project, although more advanced techniques may be used.
# Requirements
* Stitch multiple images to create a panoramic photo.
* Handle non-overlapping and multiply-overlapped images.
* Ensure the output image retains its irregular shape without cropping.
