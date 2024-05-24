selectDetections();
//Calculate Haralick features for DAB channel
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons": 0.25,  "region": "ROI",  "tileSizeMicrons": 25.0,  "colorOD": false,' +
        '  "colorStain1": true,  "colorStain2": false,  "colorStain3": false,  "colorRed": false,  "colorGreen": false,  "colorBlue": false,' +
        '  "colorHue": false,  "colorSaturation": false,  "colorBrightness": false,  "doMean": false,  "doStdDev": false,' +
        '  "doMinMax": false,  "doMedian": false,  "doHaralick": true,  "haralickDistance": 1,  "haralickBins": 32}');
// Save features
saveDetectionMeasurements('G:/Validation/Training_BG/detections/extra_features/')