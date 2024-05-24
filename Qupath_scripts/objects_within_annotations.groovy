selectedAnnotations = getAnnotationObjects().findAll{it.getPathClass() == getPathClass("Ignore")}
selectedAnnotations.each{anno->
    selectedCells = getCurrentHierarchy().getObjectsForROI(qupath.lib.objects.PathDetectionObject, anno.getROI())
}
selectedCells.each{cell->
    getCurrentHierarchy().getSelectionModel().setSelectedObject(cell, true);

}
