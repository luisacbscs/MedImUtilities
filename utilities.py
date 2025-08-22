import itk
import os
import numpy as np


# Function to read into itk.Image a DICOM series
def read_dicom_series(series_path, pixel_type=itk.F, dimension=3, print_metadata=False):
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
    dicomIO.SetLoadPrivateTags(True)
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()
    reader.UpdateLargestPossibleRegion()  # only works in function with this; only prints metadata after this

    if print_metadata:

        metadata = dicomIO.GetMetaDataDictionary()

        for tagkey in metadata.GetKeys():
            try:
                value = metadata[tagkey]
                print(tagkey, value)
            except RuntimeError:
                print("Cannot read" + tagkey + "into metadata dictionary")
            except UnicodeEncodeError:
                value = metadata[tagkey].encode('utf-8', 'surrogateescape').decode('ISO-8859-1')
                print(tagkey, value)

    return reader.GetOutput()


# Function to access a specific tag from the metadata of a DICOM series; the tag must in format ['0000', '0000'] or
# [['0000', '0000'], ['0000', '0000'], ...] if nested (with the first position corresponding to the parent tag). If no
# tag is passed, all metadata dictionary is printed.
def get_dicom_tag(series_path, dicom_tag: list = None):

    if len(os.listdir(series_path)) == 0:
        print("Empty directory!")
        return None

    import pydicom
    try:
        metadata = pydicom.filereader.dcmread(os.path.join(series_path, sorted(os.listdir(series_path))[1]))
        # reading from the second file in the series (index 1), as the first file may be a storage file (DIRFILE) and,
        # therefore, its metadata relative to DICOM series organization and storage, and not actual imaging-related tags
    except IndexError:
        metadata = pydicom.filereader.dcmread(os.path.join(series_path, sorted(os.listdir(series_path))[0]))

    if dicom_tag is not None:
        tag_name = None
        tag_value = None
        intermediate = metadata
        if np.asarray(dicom_tag).ndim == 1:
            tag1 = int(f"0x{dicom_tag[0]}", 16)
            tag2 = int(f"0x{dicom_tag[1]}", 16)

            try:
                tag_name = str(metadata[tag1, tag2].name)
                tag_value = str(metadata[tag1, tag2].value)
            except KeyError:
                print(f"Tag ({dicom_tag[0]}, {dicom_tag[1]}) not found! Check for parent tag.")

        elif np.asarray(dicom_tag).ndim == 2:
            for tag in dicom_tag:
                if tag != dicom_tag[-1]:
                    intermediate = intermediate[int(f"0x{tag[0]}", 16), int(f"0x{tag[1]}", 16)][0]
                else:
                    try:
                        tag_name = str(intermediate[int(f"0x{tag[0]}", 16), int(f"0x{tag[1]}", 16)].name)
                        tag_value = str(intermediate[int(f"0x{tag[0]}", 16), int(f"0x{tag[1]}", 16)].value)
                    except KeyError:
                        print(f"Tag ({dicom_tag[0]}, {dicom_tag[1]}) not found! Check for parent tag.")

        else:
            print("Please introduce a DICOM tag in format ['0000', '0000'], or, if nested, [['0000', '0000'],"
                  " ['0000', '0000'], ...] (hierarchically ordered from parent to child).")

        return tag_name, tag_value
    else:
        print(metadata)


# Function to resample an itk Image to a given spacing (expected spacing array format: (X, Y, Z) - ITK FORMAT)
def resample_volume(volume, new_spacing, interpolation_mode="bspline"):

    if interpolation_mode == "nearestneighbour":    # for masks
        interpolator = itk.NearestNeighborInterpolateImageFunction
    elif interpolation_mode == "linear":
        interpolator = itk.LinearInterpolateImageFunction
    else:
        interpolator = itk.BSplineInterpolateImageFunction

    if isinstance(volume, str):
        volume = itk.imread(volume)

    original_spacing = itk.spacing(volume)
    original_origin = itk.origin(volume)
    original_size = itk.size(volume)
    original_direction = volume.GetDirection()

    if original_spacing != new_spacing:

        # rounding up the new size! output physical space will be equal or larger to original
        new_size = [int(np.ceil(osz * ospc / nspc)) for osz, ospc, nspc in
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
def resample_to_reference(volume, reference, interpolation_mode="bspline"):

    if interpolation_mode == "nearestneighbour":  # for masks
        interpolator = itk.NearestNeighborInterpolateImageFunction
    elif interpolation_mode == "linear":
        interpolator = itk.LinearInterpolateImageFunction
    else:
        interpolator = itk.BSplineInterpolateImageFunction

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


# Function to read and write a DICOM series into a file
def series_reader_writer(series_path, pixel_type=itk.F, dimension=3, out_filename=None):

    ImageType = itk.Image[pixel_type, dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(series_path)

    seriesUID = namesGenerator.GetSeriesUIDs()

    if len(seriesUID) < 1:
        print("No DICOMs in: " + series_path)
        return 1

    for uid in seriesUID:

        seriesIdentifier = uid
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)
        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        writer = itk.ImageFileWriter[ImageType].New()

        if out_filename is None:
            out_filename = seriesIdentifier + ".nii.gz"

        writer.SetFileName(out_filename)
        writer.UseCompressionOn()
        writer.SetInput(reader.GetOutput())

        writer.Update()


# Function to get the coronal maximum intensity projections of a volume
def get_maximum_intensity_projections(volume, num_projections=72, progressbar=True):

    if isinstance(volume, str):
        volume = itk.imread(volume)
    elif isinstance(volume, np.ndarray):
        volume = itk.image_from_array(volume)
    elif not isinstance(volume, itk.Image):
        print("Please enter path to volume, numpy array ot itk Image!")
        return

    meta = dict(volume)
    arr = np.asarray(volume)

    angles = np.linspace(0, 360, num=num_projections, endpoint=False)

    mip_stack = np.zeros((arr.shape[0], len(angles), arr.shape[1]), dtype=arr.dtype)

    if progressbar:
        from tqdm import tqdm
        import sys
        pb = tqdm(total=len(angles), desc="Calculating projections...", file=sys.stdout)

    from scipy.ndimage import rotate
    for y, angle in enumerate(angles):
        rotated_image = rotate(arr, angle, axes=(1, 2), reshape=False, order=1)  # Rotate in-plane
        mip_proj = rotated_image.max(axis=1)  # Compute MIP along the Y-axis
        mip_stack[:, y, :] = mip_proj
        if progressbar:
            pb.update(1)

    mip_itk = itk.image_from_array(mip_stack)
    for k, v in meta.items():
        mip_itk[k] = v

    return mip_itk


def coregister_images(moving_image, fixed_image, fixed_mask=None, debug=False, get_transform=False):
    """
    :param moving_image: Moving ITK image
    :param fixed_image: Fixed ITK image
    Both images must have the same pixel type!!!! Else exit code -1073741819 (0xC0000005) (Windows access violation
     error) will be thrown
    :param pixel_type: default id ITK Double
    :param dimension: default is 3 dimensions (volume)
    :return: img1 registered to img2
    """

    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')

    # Modify the parameter map for initialization
    default_rigid_parameter_map["AutomaticTransformInitialization"] = ["true"]
    default_rigid_parameter_map["AutomaticTransformInitializationMethod"] = ["MenterOfGravity"]

    parameter_object.AddParameterMap(default_rigid_parameter_map)

    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)

    if fixed_mask is not None:
        elastix_object.SetFixedMask(fixed_mask)

    # Set additional options
    elastix_object.SetLogToConsole(debug)

    # Update filter object (required)
    elastix_object.UpdateLargestPossibleRegion()

    # Results of Registration
    registered_image = elastix_object.GetOutput()

    if get_transform:
        # Get transform parameter map
        transform_parameter_map = elastix_object.GetTransformParameterObject()
        return registered_image, transform_parameter_map

    return registered_image
