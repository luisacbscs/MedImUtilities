import itk
import os


# Function to read into itk.Image a DICOM series
def read_dicom_series(series_path, pixel_type=itk.F, dimension=3):
    # Set up the image readers with their type
    ImageType = itk.Image[pixel_type, dimension]

    # Using GDCMSeriesFileNames to generate the names of
    # DICOM files.
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetDirectory(series_path)

    # Get the names of files
    fileNames = namesGenerator.GetInputFileNames()

    # Set up the image series reader using GDCMImageIO
    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    dicomIO.LoadPrivateTagsOn()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)

    reader.UpdateLargestPossibleRegion()    # only works in function with this

    return reader.GetOutput()


# Function to access a specific tag from the metadata of a DICOM series; the tag must in format ['0000', '0000']
def get_dicom_tag(series_path, dicom_tag: list):
    import pydicom

    tag1 = int(f"0x{dicom_tag[0]}", 16)
    tag2 = int(f"0x{dicom_tag[1]}", 16)

    metadata = pydicom.filereader.dcmread(os.path.join(series_path, sorted(os.listdir(series_path))[0]))

    tag_name = str(metadata[tag1, tag2].name)
    tag_value = str(metadata[tag1, tag2].value)

    return tag_name, tag_value


# Function to resample an itk Image to a given spacing (expected spacing array format: (X, Y, Z) - ITK FORMAT)
def resample_volume(volume, new_spacing, interpolation_mode="nearestneighbour"):

    if interpolation_mode == "nearestneighbour":    # for images
        interpolator = itk.NearestNeighborInterpolateImageFunction
    elif interpolation_mode == "linear":    # for masks
        interpolator = itk.LinearInterpolateImageFunction

    if isinstance(volume, str):
        volume = itk.imread(volume)

    original_spacing = itk.spacing(volume)
    original_origin = itk.origin(volume)
    original_size = itk.size(volume)
    original_direction = volume.GetDirection()

    if original_spacing != new_spacing:

        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                    zip(original_size, original_spacing, new_spacing)]

        return itk.resample_image_filter(
            volume,
            interpolator=interpolator.New(volume),
            size=new_size,
            output_spacing=new_spacing,
            output_origin=original_origin,
            output_direction=original_direction
        )

    else:
        return volume


# Function to resample an itk Image given a reference itk Image
def resample_to_reference(volume, reference, interpolation_mode="nearestneighbour"):

    if interpolation_mode == "nearestneighbour":    # for images
        interpolator = itk.NearestNeighborInterpolateImageFunction
    elif interpolation_mode == "linear":    # for masks
        interpolator = itk.LinearInterpolateImageFunction

    if isinstance(volume, str):
        volume = itk.imread(volume)

    if isinstance(reference, str):
        reference = itk.imread(reference)

    return itk.resample_image_filter(
        volume,
        interpolator=interpolator.New(volume),
        use_reference_image=True,
        reference_image=reference
    )
