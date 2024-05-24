import qupath.lib.gui.scripting.QPEx
import qupath.lib.scripting.QP

def imagename = getCurrentServer().getPath().split('/').last().replace('.svs[--series, 0]', '')

QPEx.saveDetectionMeasurements(String.format('D:/Tanrada_classification/%s.txt', imagename))
