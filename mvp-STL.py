import vtk
import pydicom
import numpy as np

def main():
    # Set the path to the DICOM file
    path_to_dicom_file = "./CT/1.dcm"

    # Read the DICOM file
    dicom_file = pydicom.dcmread(path_to_dicom_file)
    pixel_data = dicom_file.pixel_array
    num_rows, num_cols, num_slices = np.shape(pixel_data)
    WW = dicom_file.WindowWidth
    WL = dicom_file.WindowCenter
    Rescale_intercept = dicom_file.RescaleIntercept
    Rescale_slope = dicom_file.RescaleSlope
    PixelSpacing = dicom_file.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
    SliceThickness = dicom_file.SliceThickness
    image_position = dicom_file.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient

    # Create a VTK image data object from the pixel data with high-density structures painted white
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(num_rows, num_cols, num_slices)
    vtk_image.SetSpacing((PixelSpacing[0], PixelSpacing[1], SliceThickness))
    vtk_image.SetOrigin(image_position[0], image_position[1], image_position[2])
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
    vtk_image.GetPointData().GetScalars().SetVoidArray(pixel_data.ravel(), len(pixel_data.ravel()), 1)   
    
    # actor_teeth = create_teeth_actor(vtk_image)
    actor_other = create_other_actor(vtk_image)
    actor_teeth = create_teeth_actor(vtk_image)

    colors = vtk.vtkNamedColors()

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(2)
    lut.SetTableRange(0, 1)
    lut.Build()
    lut.SetTableValue(0, colors.GetColor4d('wheat')) # mandible without teeth
    lut.SetTableValue(1, colors.GetColor4d('white'))  # teeth 

    actor_other.GetProperty().SetDiffuseColor(lut.GetTableValue(0)[:3])
    actor_other.GetProperty().SetDiffuse(1.0)
    actor_other.GetProperty().SetSpecular(0.1)
    # actor_other.GetProperty().SetSpecularPower(10000)

    actor_teeth.GetProperty().SetDiffuseColor(lut.GetTableValue(1)[:3])
    actor_teeth.GetProperty().SetDiffuse(1.0)
    actor_teeth.GetProperty().SetSpecular(0.1)
    # actor_teeth.GetProperty().SetSpecularPower(10000)        

    # Create a renderer and add the actors
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor_teeth)  
    renderer.AddActor(actor_other)      
    renderer.SetBackground(0.5, 1.0, 0.5)

    # Create a render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    interactor.Start()

def create_other_actor(vtk_image):
    # Create threshold filter for mandible without teeth
    threshold_other = vtk.vtkImageThreshold()
    threshold_other.SetInputData(vtk_image)
    threshold_other.ThresholdBetween(1700, 3000)
    # threshold_other.ReplaceInOn()
    threshold_other.SetInValue(30)
    # threshold_other.ReplaceOutOn()
    threshold_other.SetOutValue(0)
    # threshold_other.Update()

    gaussian_filter_other = vtk.vtkImageGaussianSmooth()
    gaussian_filter_other.SetInputConnection(threshold_other.GetOutputPort())
    gaussian_filter_other.SetStandardDeviations(3, 3, 3)
    gaussian_filter_other.SetRadiusFactors(3, 3, 3)

    marching_cubes_other = vtk.vtkFlyingEdges3D()
    marching_cubes_other.SetInputConnection(gaussian_filter_other.GetOutputPort())
    marching_cubes_other.ComputeScalarsOff()
    marching_cubes_other.ComputeGradientsOff()
    marching_cubes_other.ComputeNormalsOff()
    marching_cubes_other.SetValue(0, 15)

    smoother_other = vtk.vtkWindowedSincPolyDataFilter()
    smoother_other.SetInputConnection(marching_cubes_other.GetOutputPort())
    smoother_other.SetNumberOfIterations(8)
    smoother_other.BoundarySmoothingOff()
    smoother_other.FeatureEdgeSmoothingOff()
    smoother_other.SetFeatureAngle(60.0)
    smoother_other.SetPassBand(0.001)
    smoother_other.NonManifoldSmoothingOn()
    smoother_other.NormalizeCoordinatesOff()
    smoother_other.Update()

    normals_other = vtk.vtkPolyDataNormals()
    normals_other.SetInputConnection(smoother_other.GetOutputPort())
    normals_other.SetFeatureAngle(60.0)

    stripper_other = vtk.vtkStripper()
    stripper_other.SetInputConnection(normals_other.GetOutputPort())

    mapper_other = vtk.vtkPolyDataMapper()
    mapper_other.SetInputConnection(stripper_other.GetOutputPort())

    actor_other = vtk.vtkActor()
    actor_other.SetMapper(mapper_other)
    return actor_other

def create_teeth_actor(vtk_image):
    # Create threshold filter for teeth
    threshold_teeth = vtk.vtkImageThreshold()
    threshold_teeth.SetInputData(vtk_image)
    threshold_teeth.ThresholdBetween(3000, 4095)
    # threshold_teeth.ReplaceInOn()
    threshold_teeth.SetInValue(230)
    # threshold_teeth.ReplaceOutOn()
    threshold_teeth.SetOutValue(0)
    # threshold_teeth.Update()

    # Apply a Gaussian filter to smooth the binary mask
    gaussian_filter_teeth = vtk.vtkImageGaussianSmooth()
    gaussian_filter_teeth.SetInputConnection(threshold_teeth.GetOutputPort())
    gaussian_filter_teeth.SetStandardDeviations(1, 1, 1)
    gaussian_filter_teeth.SetRadiusFactors(150, 150, 150)

    # Create two marching cubes for teeth and other parts of the mandible
    marching_cubes_teeth = vtk.vtkMarchingCubes()
    marching_cubes_teeth.SetInputConnection(gaussian_filter_teeth.GetOutputPort())
    marching_cubes_teeth.ComputeScalarsOff()
    marching_cubes_teeth.ComputeGradientsOff()
    marching_cubes_teeth.ComputeNormalsOff()
    marching_cubes_teeth.SetValue(0, 0.1)

    smoother_teeth = vtk.vtkWindowedSincPolyDataFilter()
    smoother_teeth.SetInputConnection(marching_cubes_teeth.GetOutputPort())
    smoother_teeth.SetNumberOfIterations(10)
    smoother_teeth.BoundarySmoothingOff()
    smoother_teeth.FeatureEdgeSmoothingOff()
    smoother_teeth.SetFeatureAngle(60)
    smoother_teeth.SetPassBand(0.001)
    smoother_teeth.NonManifoldSmoothingOn()
    smoother_teeth.NormalizeCoordinatesOff()
    smoother_teeth.Update()

    normals_teeth = vtk.vtkPolyDataNormals()
    normals_teeth.SetInputConnection(smoother_teeth.GetOutputPort())
    normals_teeth.SetFeatureAngle(60.0)

    stripper_teeth = vtk.vtkStripper()
    stripper_teeth.SetInputConnection(normals_teeth.GetOutputPort())

    mapper_teeth = vtk.vtkPolyDataMapper()
    mapper_teeth.SetInputConnection(stripper_teeth.GetOutputPort())

    actor_teeth = vtk.vtkActor()
    actor_teeth.SetMapper(mapper_teeth)
    return actor_teeth

def create_mandible_lut(colors):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(2)
    lut.SetTableRange(0, 1)
    lut.Build()

    lut.SetTableValue(0, colors.GetColor4d('wheat')) # mandible without teeth
    lut.SetTableValue(1, colors.GetColor4d('white'))  # teeth
    return lut
    
if __name__ == '__main__':
    main()