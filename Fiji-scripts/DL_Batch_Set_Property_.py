from ij import IJ
import ij.WindowManager as WM

nonImageTitleArray = WM.getIDList()

for eachWindow in nonImageTitleArray:
    imp = WM.getImage(eachWindow)
    IJ.run(imp, "Properties...", "channels=1 slices=1 frames=1 unit=micron pixel_width=0.0300000 pixel_height=0.0300000 voxel_depth=1.0000000");
