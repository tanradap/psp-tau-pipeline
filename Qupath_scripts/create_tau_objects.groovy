// clear out old stuff
selectDetections()
clearSelectedObjects(true);

// select only relevant annotations & run cell detection
selectObjectsByClassification("STN_reseg", "GP_reseg","STR_reseg");
createDetectionsFromPixelClassifier("tau_high_s0_t0.25", 5.0, 0.0, "SPLIT")

// Calculate measurements
selectDetections()
addShapeMeasurements("AREA", "LENGTH", "CIRCULARITY", "SOLIDITY", "MAX_DIAMETER", "MIN_DIAMETER", "NUCLEUS_CELL_RATIO")
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons": 0.25,  "region": "ROI",  "tileSizeMicrons": 25.0,  "colorOD": false,  "colorStain1": true,  "colorStain2": true,  "colorStain3": false,  "colorRed": true,  "colorGreen": true,  "colorBlue": true,  "colorHue": false,  "colorSaturation": true,  "colorBrightness": true,  "doMean": true,  "doStdDev": true,  "doMinMax": true,  "doMedian": true,  "doHaralick": false,  "haralickDistance": 1,  "haralickBins": 32}');
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons": 0.25,  "region": "ROI",  "tileSizeMicrons": 25.0,  "colorOD": false,  "colorStain1": false,  "colorStain2": true,  "colorStain3": false,  "colorRed": false,  "colorGreen": false,  "colorBlue": false,  "colorHue": false,  "colorSaturation": false,  "colorBrightness": false,  "doMean": false,  "doStdDev": false,  "doMinMax": false,  "doMedian": false,  "doHaralick": true,  "haralickDistance": 1,  "haralickBins": 32}');

// Save features
saveDetectionMeasurements('D:/Validation/Training_BG/detections_noartefact/')
