import qupath.lib.gui.scripting.QPEx
import qupath.lib.images.servers.ImageServer
import qupath.lib.scripting.QP
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.RegionRequest
import qupath.lib.images.servers.LabeledImageServer
import java.lang.Math

import java.awt.image.BufferedImage

// Write the full image (only possible if it isn't too large!)
//def P_SIZE = 0.2528
def cal = getCurrentServer().getPixelCalibration()
double pixelWidth = cal.getPixelWidthMicrons()
double pixelHeight = cal.getPixelHeightMicrons()

def delta_x = 0.1/pixelWidth
def delta_y = 0.1/pixelHeight

def imageData = QP.getCurrentImageData()
def server = QP.getCurrentServer()
def viewer = QPEx.getCurrentViewer()
def imagename = server.getPath().split('/').last().replace('.svs', '').replace('[--series, 0]', '')
def detections = "D:/Validation/Training_BG/detections_noartefact/predictions"
def cell_info = detections + '/' + imagename + '.svs_predictions_NEW.txt'
def linenum = 0

// Reading in values from prediction.txt file & storing in arraylists (x, y, c)
def is = new Scanner(new File(cell_info))
ArrayList<Double> cells_x = new ArrayList<Double>();
ArrayList<Double> cells_y = new ArrayList<Double>();
ArrayList<String> cells_c = new ArrayList<String>();
while (is.hasNextLine()) {
    acell = is.nextLine()
    if (linenum > 0) {
        cells_x.add(Double.parseDouble(acell.split('\t')[4])/pixelWidth) //convert to micron/pixel
        cells_y.add(Double.parseDouble(acell.split('\t')[5])/pixelHeight)
        cells_c.add(acell.split('\t')[3])
    }
    linenum++;
}

//number of cells (size of arraylist containing x coordinates)
def cellnum = cells_x.size()

// Get detection measurements: for each detection,
QP.getDetectionObjects().each { detection -> //Get x,y coordinates of each detection
    xc = detection.getROI().getCentroidX() //already in micron/pixel
    yc = detection.getROI().getCentroidY()
    //println("xc "+ xc)
    //println("yc "+ yc)
    if (!(xc.isNaN())) { //if x coordinate is not NAN - not empty
        //flag = 0
        i = 0
        cellnum = cells_x.size() //get size of candidate pool (text file predictions)
        while (i < cellnum) { // go through each cell in candidate pool
            diff_x = Math.abs(xc-cells_x[i])
            diff_y = Math.abs(yc-cells_y[i])
            if ((diff_x <delta_x) && (diff_y <delta_y)) { //search for the match: for each qupath detection, go through cells in text file input
                detection.setPathClass(QP.getPathClass(cells_c[i])) //when found, assign class
                //flag = 1
               // println("diff_x " + diff_x)
               // println("diff_y " + diff_y )
               // println("cell x " +  cells_x[i])
                //println("cell y " + cells_y[i])

                cells_x.remove(i)
                cells_y.remove(i)
                cells_c.remove(i)
                break // ends the search
            }
            i++
        }

    }
}
