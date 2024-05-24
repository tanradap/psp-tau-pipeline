import qupath.imagej.objects.*

def imageData = getCurrentImageData()
def pathObjects = getSelectedObjects()
def cal = getCurrentServer().getPixelCalibration()
double pixelWidth = cal.getPixelWidthMicrons()
double pixelHeight = cal.getPixelHeightMicrons()

hierarchy = getCurrentHierarchy()
getDetectionObjects().each{
    def ml = it.getMeasurementList()
    def roi = it.getROI()
    
    ml.putMeasurement('Centroid X',roi.getCentroidX()*pixelWidth)
    ml.putMeasurement('Centroid Y',roi.getCentroidY()*pixelHeight)
    ml.close()
}
fireHierarchyUpdate()