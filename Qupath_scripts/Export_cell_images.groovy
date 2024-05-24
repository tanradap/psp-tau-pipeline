import qupath.lib.gui.scripting.QPEx
import qupath.lib.images.servers.ImageServer
import qupath.lib.scripting.QP
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.RegionRequest
import qupath.lib.images.servers.LabeledImageServer

import java.awt.image.BufferedImage

// Write the full image (only possible if it isn't too large!)
def P_SIZE = 0.2528
def imageData = QP.getCurrentImageData()
def server = QP.getCurrentServer()
def viewer = QPEx.getCurrentViewer()
def imagename = server.getPath().split('/').last().replace('.svs', '').replace('[--series, 0]', '')
def detections = "D:/Classification/"
def cell_info = detections + '/' + imagename + '_incorrect.txt'
def linenum = 0
def is = new Scanner(new File(cell_info))

ArrayList<Double> cells_x = new ArrayList<Double>();
ArrayList<Double> cells_y = new ArrayList<Double>();
while(is.hasNextLine())
{
    acell = is.nextLine()
    if (linenum > 0) {
        cells_x.add(Double.parseDouble(acell.split('\t')[2])/P_SIZE)
        cells_y.add(Double.parseDouble(acell.split('\t')[3])/P_SIZE)
    }
    linenum++;
}

def cellnum = cells_x.size()
def i = 0
def windowsize = 100
def save_path ="D:/Tanrada_classification/"
while (i < cellnum) {
    plane = ImagePlane.getPlane(0, 0)
    xroi = Math.round(cells_x[i]-windowsize/2)
    yroi = Math.round(cells_y[i]-windowsize/2)
    roi = ROIs.createRectangleROI(xroi, yroi, windowsize, windowsize, plane)
    requestROI = RegionRequest.createInstance(server.getPath(), 1, roi)
//    ImageWriterTools.writeImageRegionWithOverlay(img, imageData, overlayOptions, request, fileImageWithOverlay.getAbsolutePath())
    QPEx.writeRenderedImageRegion(viewer, requestROI, save_path + String.format('/cells/%s_%s.tif', imagename, i.toString()))
    i++
}
