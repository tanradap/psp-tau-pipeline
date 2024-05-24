selectedAnnotations = getAnnotationObjects().findAll{it.getPathClass() == getPathClass("GP")}
selectedAnnotations.each{anno->
    selectedCells = getCurrentHierarchy().getObjectsForROI(qupath.lib.objects.PathDetectionObject, anno.getROI())
}
selectedCells.each{cell->
    cell.setName("Globus Pallidus")

}
