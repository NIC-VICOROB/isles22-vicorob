import SimpleITK as sitk
import copy

import numpy as np

def match_histogram_sitk(ref, mov, matched_filepath=None, hist_levels=256, match_points=12, mean_thresh=True):
    """Performs MRI histogram matching [#f1]_ using Simple ITK.
    .. rubric:: Footnotes
    .. [#f1] Laszlo G. Nyul, Jayaram K. Udupa, and Xuan Zhang, "New Variants of a Method of MRI Scale Standardization", IEEE Transactions on Medical Imaging, 19(2):143-150, 2000.
    :param ref: Sitk image or Filepath of nifti image for reference histogram.
    :param mov: Sitk image or Filepath of nifti image to transform.
    :param matched_filepath: Filepath of output image (input with histogram matched to reference).
    """
    if isinstance(ref, str):
        ref = sitk.ReadImage(ref, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")
    if isinstance(mov, str):
        mov = sitk.ReadImage(mov, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(hist_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(mean_thresh)
    enhanced_mov = matcher.Execute(mov, ref)

    if matched_filepath is not None:
        sitk.WriteImage(enhanced_mov, matched_filepath)

    return enhanced_mov


def register_affine_sitk(
        ref,
        mov,
        transform_fpath=None,
        warped_fpath=None,
        warp_interpolation=sitk.sitkLinear,
        seed=1,
        num_threads=1,
        do_histogram_matching=True,
        optimizer_iterations=(2100, 1000, 1000, 10),
        smoothing_sigmas=(3, 2, 1, 0),
        shrink_factors=(6, 4, 2, 1),
        optimizer_learning_rate=0.25,
        optimizer_convergence_min_value=1e-6,
        optimizer_convergence_win_size=10,
        metric_name='MI',
        metric_number_bins=32,
        metric_sampling_percentage=0.2,
        verbose=False
):
    # convert np arrays into itk image objects
    if isinstance(ref, str):
        ref = sitk.ReadImage(ref, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")
    if isinstance(mov, str):
        mov = sitk.ReadImage(mov, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")

    # Store mov_original for later
    mov_original = copy.deepcopy(mov)
    
    ### PREPROCESS IMAGES
    def preprocess_image(img):
        minmax_filter = sitk.MinimumMaximumImageFilter()
        minmax_filter.Execute(img)
        intensity_windower = sitk.IntensityWindowingImageFilter()
        intensity_windower.SetWindowMinimum(minmax_filter.GetMinimum())
        intensity_windower.SetWindowMaximum(minmax_filter.GetMaximum())
        intensity_windower.SetOutputMinimum(0.0)
        intensity_windower.SetOutputMaximum(1.0)
        return intensity_windower.Execute(img)
    ref = preprocess_image(ref)
    mov = preprocess_image(mov)
    
    if do_histogram_matching:
        mov = match_histogram_sitk(ref=ref, mov=mov)
    
    # Registration parameters
    registration = sitk.ImageRegistrationMethod()
    registration.SetGlobalDefaultNumberOfThreads(num_threads)
    
    # Similarity metric settings
    if metric_name == 'MI':
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=metric_number_bins)
    elif metric_name == 'CC':
        registration.SetMetricAsCorrelation()
    else:
        raise ValueError(f'Given metric_name {metric_name} not one of ("MI", "CC")')
    
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(metric_sampling_percentage, seed)
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings
    registration.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=optimizer_learning_rate,
            numberOfIterations=max(optimizer_iterations), # Set biggest num of iters, and later stop at right one
            convergenceMinimumValue=optimizer_convergence_min_value,
            convergenceWindowSize=optimizer_convergence_win_size,
            lineSearchLowerLimit=0.0,
            lineSearchUpperLimit=2.0,
            lineSearchEpsilon=0.2,
            lineSearchMaximumIterations=20,
            maximumStepSizeInPhysicalUnits=optimizer_learning_rate,# / 4.0,
            estimateLearningRate=registration.EachIteration)
    registration.SetOptimizerScalesFromPhysicalShift()
    
    registration.MetricUseFixedImageGradientFilterOff()
    registration.MetricUseMovingImageGradientFilterOff()
    
    def check_iteration(method: sitk.ImageRegistrationMethod):
        if verbose:
            print(f"Level {method.GetCurrentLevel()} Iter {method.GetOptimizerIteration()} = " + \
                  f"Metric {method.GetMetricValue():10.10e} : ConvergenceValue {method.GetOptimizerConvergenceValue():10.10e}")
        
        if method.GetOptimizerIteration() >= optimizer_iterations[method.GetCurrentLevel()]:
            method.StopRegistration()  # It stops the current level, not the WHOLE registration
        
    registration.AddCommand(sitk.sitkIterationEvent, lambda: check_iteration(registration))
    
    # Setup for the multi-resolution framework.
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    # Set initial transform for moving
    initial_tf = sitk.CenteredTransformInitializer(ref,
                                                   mov,
                                                   sitk.AffineTransform(3),
                                                   sitk.CenteredTransformInitializerFilter.MOMENTS)
    
    # Initialize affine transform and optimize
    registration.SetInitialTransform(initial_tf, inPlace=True)
    
    ### EXECUTE REGISTRATION
    affine_tf = registration.Execute(ref, mov)
    if verbose:
        print(affine_tf)
    
    # Store outputs
    if transform_fpath is not None:
        sitk.WriteTransform(affine_tf, transform_fpath)
    if warped_fpath is not None:
        transform_sitk_image(
                mov=mov_original,
                ref=ref,
                transform=affine_tf,
                warped_fpath=warped_fpath,
                interpolator=warp_interpolation)
    
    return affine_tf


def transform_sitk_image(mov, ref, transform, warped_fpath=None, interpolator=sitk.sitkLinear, default_value=0.0):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    if isinstance(ref, str):
        ref = sitk.ReadImage(ref, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")
    if isinstance(mov, str):
        mov = sitk.ReadImage(mov, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")
    if isinstance(transform, str):
        transform = sitk.ReadTransform(transform)

    mov_warped = sitk.Resample(mov, ref, transform, interpolator, default_value)

    # Clamp just in case there were interpolated values outside of the original mov range
    minmax_filter = sitk.MinimumMaximumImageFilter()
    minmax_filter.Execute(mov)

    mov_warped = sitk.Clamp(
        image1=mov_warped, 
        outputPixelType=sitk.sitkFloat32, 
        lowerBound=min(float(minmax_filter.GetMinimum()), 0.0), 
        upperBound=float(minmax_filter.GetMaximum()))
    
    if warped_fpath is not None:
        sitk.WriteImage(mov_warped, warped_fpath)
        
    return mov_warped

def resample_spacing_sitk_image(mov, ref, warped_fpath=None, interpolator=sitk.sitkLinear, default_value=0.0):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    if isinstance(ref, str):
        ref = sitk.ReadImage(ref, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")
    if isinstance(mov, str):
        mov = sitk.ReadImage(mov, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")

    resample = sitk.ResampleImageFilter()
    #resample.SetNumberOfThreads(1)
    #resample.SetGlobalDefaultNumberOfThreads(1)
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(ref.GetDirection())
    resample.SetOutputOrigin(ref.GetOrigin())

    resample.SetOutputSpacing(ref.GetSpacing())
    resample.SetSize(ref.GetSize()) #orig_size = np.array(image.GetSize(), dtype=np.int)

    mov_warped = resample.Execute(mov)

    # Clamp just in case there were interpolated values outside of the original mov range
    minmax_filter = sitk.MinimumMaximumImageFilter()
    minmax_filter.Execute(mov)

    mov_warped = sitk.Clamp(
        image1=mov_warped, 
        outputPixelType=sitk.sitkFloat32, 
        lowerBound=min(float(minmax_filter.GetMinimum()), 0.0), 
        upperBound=float(minmax_filter.GetMaximum()))
    
    if warped_fpath is not None:
        sitk.WriteImage(mov_warped, warped_fpath)
        
    return mov_warped


def change_spacing_sitk_image(img, out_spacing, warped_fpath=None, interpolator=sitk.sitkLinear):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    if isinstance(img, str):
        img = sitk.ReadImage(img, outputPixelType=sitk.sitkFloat32, imageIO="NiftiImageIO")
   
    resample = sitk.ResampleImageFilter()
    #resample.SetGlobalDefaultNumberOfThreads(1)
    #resample.SetNumberOfThreads(1)

    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size) #orig_size = np.array(image.GetSize(), dtype=np.int)

    img_warped = resample.Execute(img)

    # Clamp just in case there were interpolated values outside of the original mov range
    minmax_filter = sitk.MinimumMaximumImageFilter()
    minmax_filter.Execute(img)

    img_warped = sitk.Clamp(
        image1=img_warped, 
        outputPixelType=sitk.sitkFloat32, 
        lowerBound=min(float(minmax_filter.GetMinimum()), 0.0), 
        upperBound=float(minmax_filter.GetMaximum()))
    
    if warped_fpath is not None:
        sitk.WriteImage(img_warped, warped_fpath)
        
    return img_warped