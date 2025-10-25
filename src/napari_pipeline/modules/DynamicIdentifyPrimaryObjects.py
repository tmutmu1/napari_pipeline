import math

import centrosome.cpmorphology
import centrosome.outline
import centrosome.propagate
import centrosome.threshold
import numpy
import scipy.ndimage
import scipy.sparse
import skimage.morphology
import skimage.segmentation
from napari_pipeline.modules.Threshold import Threshold
from napari_pipeline.CellProfilerObjectsWrapper import Objects


import os
import json
import textwrap
import re

__doc__ = """\
IdentifyPrimaryObjects
======================

**IdentifyPrimaryObjects** identifies biological objects of interest.
It requires grayscale images containing bright objects on a dark background.
Incoming images must be 2D (including 2D slices of 3D images);
please use the **Watershed** module for identification of objects in 3D.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **IdentifySecondaryObjects**, **IdentifyTertiaryObjects**,
**IdentifyObjectsManually**, and **Watershed** (for segmentation of 3D objects).

What is a primary object?
^^^^^^^^^^^^^^^^^^^^^^^^^

{DEFINITION_OBJECT}

We define an object as *primary* when it can be found in an image without needing the
assistance of another cellular feature as a reference. For example:

-  The nuclei of cells are usually more easily identifiable than whole-
   cell stains due to their
   more uniform morphology, high contrast relative to the background
   when stained, and good separation between adjacent nuclei. These
   qualities typically make them appropriate candidates for primary
   object identification.
-  In contrast, whole-cell stains often yield irregular intensity patterns
   and are lower-contrast with more diffuse staining, making them more
   challenging to identify than nuclei without some supplemental image
   information being provided. In addition, cells often touch or even overlap
   their neighbors making it harder to delineate the cell borders. For
   these reasons, cell bodies are better suited for *secondary object*
   identification, because they are best identified by using a
   previously-identified primary object (i.e, the nuclei) as a
   reference. See the **IdentifySecondaryObjects** module for details on
   how to do this.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

To use this module, you will need to make sure that your input image has
the following qualities:

-  The image should be grayscale.
-  The foreground (i.e, regions of interest) are lighter than the
   background.
-  The image should be 2D. 2D slices of 3D images are acceptable if the
   image has not been loaded as volumetric in the **NamesAndTypes**
   module. For volumetric analysis
   of 3D images, please see the **Watershed** module.

If this is not the case, other modules can be used to pre-process the
images to ensure they are in the proper form:

-  If the objects in your images are dark on a light background, you
   should invert the images using the Invert operation in the
   **ImageMath** module.
-  If you are working with color images, they must first be converted to
   grayscale using the **ColorToGray** module.
-  If your images are brightfield/phase/DIC, they may be processed with the
   **EnhanceOrSuppressFeatures** module with its "*Texture*" or "*DIC*" settings.
-  If you struggle to find effective settings for this module, you may
   want to check our `tutorial`_ on preprocessing these images with
   ilastik prior to using them in CellProfiler.

What are the advanced settings?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**IdentifyPrimaryObjects** allows you to tweak your settings in many ways;
so many that it can often become confusing where you should start. This is
typically the most important but complex step in creating a good pipeline,
so do not be discouraged: other modules are easier to configure!
Using **IdentifyPrimaryObjects** with *'Use advanced settings?'* set to *'No'*
allows you to quickly try to identify your objects based only their typical size;
CellProfiler will then use its built-in defaults to decide how to set the
threshold and how to break clumped objects apart. If you are happy with the
results produced by the default settings, you can then move on to
construct the rest of your pipeline; if not, you can set
*'Use advanced settings?'* to *'Yes'* which will allow you to fully tweak and
customize all the settings.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

A set of primary objects are produced by this module, which can be used
in downstream modules for measurement purposes or other operations. See
the section "Measurements made by this module" below
for the measurements that are produced directly by this module. Once the module
has finished processing, the module display window will show the
following panels:

-  *Upper left:* The raw, original image.
-  *Upper right:* The identified objects shown as a color image where
   connected pixels that belong to the same object are assigned the same
   color (*label image*). Note that assigned colors
   are arbitrary; they are used simply to help you distinguish the
   various objects.
-  *Lower left:* The raw image overlaid with the colored outlines of the
   identified objects. Each object is assigned one of three (default)
   colors:

   -  Green: Acceptable; passed all criteria
   -  Magenta: Discarded based on size
   -  Yellow: Discarded due to touching the border

   If you need to change the color defaults, you can make adjustments in
   *File > Preferences*.
-  *Lower right:* A table showing some of the settings used by the module
   in order to produce the objects shown. Some of these are as you
   specified in settings; others are calculated by the module itself.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* The number of primary objects identified.
-  *OriginalThreshold:* The global threshold for the image.
-  *FinalThreshold:* For the global threshold methods, this value is the
   same as *OriginalThreshold*. For the adaptive or per-object methods,
   this value is the mean of the local thresholds.
-  *WeightedVariance:* The sum of the log-transformed variances of the
   foreground and background pixels, weighted by the number of pixels in
   each distribution.
-  *SumOfEntropies:* The sum of entropies computed from the foreground
   and background distributions.

**Object measurements:**

-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the
   primary object centroids. The centroid is calculated as the center of
   mass of the binary representation of the object.

Technical notes
^^^^^^^^^^^^^^^

CellProfiler contains a modular three-step strategy to identify objects
even if they touch each other ("declumping"). It is based on previously
published
algorithms (*Malpica et al., 1997; Meyer and Beucher, 1990; Ortiz de
Solorzano et al., 1999; Wahlby, 2003; Wahlby et al., 2004*). Choosing
different options for each of these three steps allows CellProfiler to
flexibly analyze a variety of different types of objects. The module has
many options, which vary in terms of speed and sophistication. More
detail can be found in the Settings section below. Here are the three
steps, using an example where nuclei are the primary objects:

#. CellProfiler determines whether a foreground region is an individual
   nucleus or two or more clumped nuclei.
#. The edges of nuclei are identified, using thresholding if the object
   is a single, isolated nucleus, and using more advanced options if the
   object is actually two or more nuclei that touch each other.
#. Some identified objects are discarded or merged together if they fail
   to meet certain your specified criteria. For example, partial objects
   at the border of the image can be discarded, and small objects can be
   discarded or merged with nearby larger ones. A separate module,
   **FilterObjects**, can further refine the identified nuclei, if
   desired, by excluding objects that are a particular size, shape,
   intensity, or texture.

References
^^^^^^^^^^

-  Malpica N, de Solorzano CO, Vaquero JJ, Santos, A, Vallcorba I,
   Garcia-Sagredo JM, del Pozo F (1997) “Applying watershed algorithms
   to the segmentation of clustered nuclei.” *Cytometry* 28, 289-297.
   (`link`_)
-  Meyer F, Beucher S (1990) “Morphological segmentation.” *J Visual
   Communication and Image Representation* 1, 21-46.
   (`link <https://doi.org/10.1016/1047-3203(90)90014-M>`__)
-  Ortiz de Solorzano C, Rodriguez EG, Jones A, Pinkel D, Gray JW, Sudar
   D, Lockett SJ. (1999) “Segmentation of confocal microscope images of
   cell nuclei in thick tissue sections.” *Journal of Microscopy-Oxford*
   193, 212-226.
   (`link <https://doi.org/10.1046/j.1365-2818.1999.00463.x>`__)
-  Wählby C (2003) *Algorithms for applied digital image cytometry*,
   Ph.D., Uppsala University, Uppsala.
-  Wählby C, Sintorn IM, Erlandsson F, Borgefors G, Bengtsson E. (2004)
   “Combining intensity, edge and shape information for 2D and 3D
   segmentation of cell nuclei in tissue sections.” *J Microsc* 215,
   67-76.
   (`link <https://doi.org/10.1111/j.0022-2720.2004.01338.x>`__)

.. _link: https://doi.org/10.1002/(SICI)1097-0320(19970801)28:4%3C289::AID-CYTO3%3E3.0.CO;2-7
.. _tutorial: http://blog.cellprofiler.org/2017/01/19/cellprofiler-ilastik-superpowered-segmentation/

"""


#################################################
#
# Ancient offsets into the settings for Matlab pipelines
#
#################################################
IMAGE_NAME_VAR = 0
OBJECT_NAME_VAR = 1
SIZE_RANGE_VAR = 2
EXCLUDE_SIZE_VAR = 3
MERGE_CHOICE_VAR = 4
EXCLUDE_BORDER_OBJECTS_VAR = 5
THRESHOLD_METHOD_VAR = 6
THRESHOLD_CORRECTION_VAR = 7
THRESHOLD_RANGE_VAR = 8
OBJECT_FRACTION_VAR = 9
UNCLUMP_METHOD_VAR = 10
WATERSHED_VAR = 11
SMOOTHING_SIZE_VAR = 12
MAXIMA_SUPPRESSION_SIZE_VAR = 13
LOW_RES_MAXIMA_VAR = 14
SAVE_OUTLINES_VAR = 15
FILL_HOLES_OPTION_VAR = 16
TEST_MODE_VAR = 17
AUTOMATIC_SMOOTHING_VAR = 18
AUTOMATIC_MAXIMA_SUPPRESSION = 19
MANUAL_THRESHOLD_VAR = 20
BINARY_IMAGE_VAR = 21
MEASUREMENT_THRESHOLD_VAR = 22

#################################################
#
# V10 introduced a more unified handling of
#     threshold settings.
#
#################################################
OFF_THRESHOLD_METHOD_V9 = 6
OFF_THRESHOLD_CORRECTION_V9 = 7
OFF_THRESHOLD_RANGE_V9 = 8
OFF_OBJECT_FRACTION_V9 = 9
OFF_MANUAL_THRESHOLD_V9 = 19
OFF_BINARY_IMAGE_V9 = 20
OFF_TWO_CLASS_OTSU_V9 = 24
OFF_USE_WEIGHTED_VARIANCE_V9 = 25
OFF_ASSIGN_MIDDLE_TO_FOREGROUND_V9 = 26
OFF_THRESHOLDING_MEASUREMENT_V9 = 31
OFF_ADAPTIVE_WINDOW_METHOD_V9 = 32
OFF_ADAPTIVE_WINDOW_SIZE_V9 = 33
OFF_FILL_HOLES_V10 = 12
OFF_N_SETTINGS = 16

"""The number of settings, exclusive of threshold settings"""
N_SETTINGS = 19

UN_INTENSITY = "Intensity"
UN_SHAPE = "Shape"
UN_LOG = "Laplacian of Gaussian"
UN_NONE = "None"

WA_INTENSITY = "Intensity"
WA_SHAPE = "Shape"
WA_PROPAGATE = "Propagate"
WA_NONE = "None"

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

DEFAULT_MAXIMA_COLOR = "Blue"

"""Never fill holes"""
FH_NEVER = "Never"
FH_THRESHOLDING = "After both thresholding and declumping"
FH_DECLUMP = "After declumping only"

FH_ALL = (FH_NEVER, FH_THRESHOLDING, FH_DECLUMP)

# Settings text which is referenced in various places in the help
SIZE_RANGE_SETTING_TEXT = "Typical diameter of objects, in pixel units (Min,Max)"
EXCLUDE_SIZE_SETTING_TEXT = "Discard objects outside the diameter range?"
AUTOMATIC_SMOOTHING_SETTING_TEXT = (
    "Automatically calculate size of smoothing filter for declumping?"
)
SMOOTHING_FILTER_SIZE_SETTING_TEXT = "Size of smoothing filter"
AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT = (
    "Automatically calculate minimum allowed distance between local maxima?"
)

# Icons for use in the help


class DynamicIdentifyPrimaryObjects(
    # cellprofiler_core.module.image_segmentation.ImageSegmentation
):
    variable_revision_number = 15

    category = "Object Processing"

    module_name = "DynamicIdentifyPrimaryObjects"

    def __init__(self):
        self.threshold = Threshold()
        self.disabled = False


    def volumetric(self):
        return False


    def visible_settings(self):
        visible_settings = [self.settings_text_file, self.get_settings_button, self.use_advanced]

        # visible_settings += super(DynamicIdentifyPrimaryObjects, self).visible_settings()
        visible_settings += [self.x_name, self.y_name]

        visible_settings += [
            self.size_range,
            self.exclude_size,
            self.exclude_border_objects,
        ]

        if self.use_advanced:
            visible_settings += self.threshold.visible_settings()[2:]

            visible_settings += [self.unclump_method, self.watershed_method]

            if self.unclump_method != UN_NONE and self.watershed_method != WA_NONE:
                visible_settings += [self.automatic_smoothing]

                if not self.automatic_smoothing:
                    visible_settings += [self.smoothing_filter_size]

                visible_settings += [self.automatic_suppression]

                if not self.automatic_suppression:
                    visible_settings += [self.maxima_suppression_size]

                visible_settings += [self.low_res_maxima, self.want_plot_maxima]

                if self.want_plot_maxima:
                    visible_settings += [self.maxima_color, self.maxima_size]

            else:  # self.unclump_method == UN_NONE or self.watershed_method == WA_NONE
                visible_settings = visible_settings[:-2]

                if self.unclump_method == UN_NONE:
                    visible_settings += [self.unclump_method]
                else:  # self.watershed_method == WA_NONE
                    visible_settings += [self.watershed_method]

            visible_settings += [self.fill_holes, self.limit_choice]

            if self.limit_choice != LIMIT_NONE:
                visible_settings += [self.maximum_object_count]
        
        return visible_settings

    # Get dynamic settings based on input image
    def get_dynamic_settings(self, settingsDict):       
        for key, value in settingsDict.items():
            if (key.startswith("threshold.")):
                setattr(self.threshold, key[len("threshold."):], value)
            else:
                setattr(self, key, value)
        # markerName = re.search("[^\W_]+(?=_[^\W_]+\.[^\W_]+$)", input_image.file_name)
        # if (markerName is not None):
        #     markerSettings = settingsDict.get(markerName.group().lower())
        #     if (markerSettings is not None):
        #         self.size_range = markerSettings.get("size_range")
        #         self.exclude_size = markerSettings.get("exclude_size")
        #         self.exclude_border_objects = markerSettings.get("exclude_border_objects")
        #         self.unclump_method = markerSettings.get("unclump_method")
        #         self.watershed_method = markerSettings.get("watershed_method")
        #         self.smoothing_filter_size = markerSettings.get("smoothing_filter_size")
        #         self.maxima_suppression_size = markerSettings.get("maxima_suppression_size")
        #         self.low_res_maxima = markerSettings.get("low_res_maxima")
        #         self.fill_holes = markerSettings.get("fill_holes")
        #         self.automatic_smoothing = markerSettings.get("automatic_smoothing")
        #         self.automatic_suppression = markerSettings.get("automatic_suppression")
        #         self.limit_choice = markerSettings.get("limit_choice")
        #         self.maximum_object_count = markerSettings.get("maximum_object_count")
        #         self.use_advanced = markerSettings.get("use_advanced")
        #         self.threshold.threshold_scope = markerSettings.get("threshold.threshold_scope")
        #         self.threshold.global_operation = markerSettings.get("threshold.global_operation")
        #         self.threshold.threshold_smoothing_scale = markerSettings.get("threshold.threshold_smoothing_scale")
        #         self.threshold.threshold_correction_factor = markerSettings.get("threshold.threshold_correction_factor")
        #         self.threshold.threshold_range = markerSettings.get("threshold.threshold_range")
        #         self.threshold.manual_threshold = markerSettings.get("threshold.manual_threshold")
        #         self.threshold.thresholding_measurement = markerSettings.get("threshold.thresholding_measurement")
        #         self.threshold.two_class_otsu = markerSettings.get("threshold.two_class_otsu")
        #         self.threshold.log_transform = markerSettings.get("threshold.log_transform")
        #         self.threshold.assign_middle_to_foreground = markerSettings.get("threshold.assign_middle_to_foreground")
        #         self.threshold.adaptive_window_size = markerSettings.get("threshold.adaptive_window_size")
        #         self.threshold.lower_outlier_fraction = markerSettings.get("threshold.lower_outlier_fraction")
        #         self.threshold.upper_outlier_fraction = markerSettings.get("threshold.upper_outlier_fraction")
        #         self.threshold.averaging_method = markerSettings.get("threshold.averaging_method")
        #         self.threshold.variance_method = markerSettings.get("threshold.variance_method")
        #         self.threshold.number_of_deviations = markerSettings.get("threshold.number_of_deviations")
        #         self.threshold.local_operation = markerSettings.get("threshold.local_operation")
        #     else:
        #         raise ValueError("Error: Marker \"" + markerName.group() + "\" does not have an entry in the settings file")
        # else:
        #     raise ValueError("Error: A marker name could not be found for input image " + input_image.file_name)

    @property
    def advanced(self):
        return self.use_advanced

    @property
    def basic(self):
        return not self.advanced

    def run(self, input_image, settingsDict, workspace):
        # workspace.display_data.statistics = []

        # input_image = workspace.image_set.get_image(
        #     self.x_name, must_be_grayscale=True
        # )
        self.y_name = input_image.file_name
        self.get_dynamic_settings(settingsDict)

        final_threshold, orig_threshold, guide_threshold, binary_image, sigma = self.threshold.get_threshold(
            input_image, workspace, automatic=self.basic
        )

        self.threshold.add_threshold_measurements(
            self.y_name,
            workspace.measurements,
            final_threshold,
            orig_threshold,
            guide_threshold,
        )

        self.threshold.add_fg_bg_measurements(
            self.y_name, workspace.measurements, input_image, binary_image
        )

        global_threshold = numpy.mean(numpy.atleast_1d(final_threshold))

        #
        # Fill background holes inside foreground objects
        #
        def size_fn(size, is_foreground):
            return size < self.size_range[1] * self.size_range[1]

        if self.basic or self.fill_holes == FH_THRESHOLDING:
            binary_image = centrosome.cpmorphology.fill_labeled_holes(
                binary_image, size_fn=size_fn
            )

        labeled_image, object_count = scipy.ndimage.label(
            binary_image, numpy.ones((3, 3), bool)
        )

        (
            labeled_image,
            object_count,
            maxima_suppression_size,
        ) = self.separate_neighboring_objects(workspace, input_image, labeled_image, object_count)

        unedited_labels = labeled_image.copy()

        # Filter out objects touching the border or mask
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = self.filter_on_border(input_image, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0

        # Filter out small and large objects
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = self.filter_on_size(
            labeled_image, object_count
        )
        size_excluded_labeled_image[labeled_image > 0] = 0

        #
        # Fill holes again after watershed
        #
        if self.basic or self.fill_holes != FH_NEVER:
            labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)

        # Relabel the image
        labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)

        if self.advanced and self.limit_choice == LIMIT_ERASE:
            if object_count > self.maximum_object_count:
                labeled_image = numpy.zeros(labeled_image.shape, int)
                border_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
                size_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
                object_count = 0

        # Make an outline image
        outline_image = centrosome.outline.outline(labeled_image)
        outline_size_excluded_image = centrosome.outline.outline(
            size_excluded_labeled_image
        )
        outline_border_excluded_image = centrosome.outline.outline(
            border_excluded_labeled_image
        )

        # Add image measurements
        objname = self.y_name
        measurements = workspace.measurements
        
        # Add label matrices to the object set
        objects = Objects()
        objects.segmented = labeled_image
        objects.unedited_segmented = unedited_labels
        objects.small_removed_segmented = small_removed_labels
        objects.parent_image = input_image

        return objects

        # workspace.object_set.add_objects(objects, self.y_name)

        # self.add_measurements(workspace)

    def smooth_image(self, image, mask):
        """Apply the smoothing filter to the image"""

        filter_size = self.calc_smoothing_filter_size()
        if filter_size == 0:
            return image
        sigma = filter_size / 2.35
        #
        # We not only want to smooth using a Gaussian, but we want to limit
        # the spread of the smoothing to 2 SD, partly to make things happen
        # locally, partly to make things run faster, partly to try to match
        # the Matlab behavior.
        #
        filter_size = max(int(float(filter_size) / 2.0), 1)
        f = (
            1
            / numpy.sqrt(2.0 * numpy.pi)
            / sigma
            * numpy.exp(
                -0.5 * numpy.arange(-filter_size, filter_size + 1) ** 2 / sigma ** 2
            )
        )

        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f, axis=0, mode="constant")
            return scipy.ndimage.convolve1d(output, f, axis=1, mode="constant")

        #
        # Use the trick where you similarly convolve an array of ones to find
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        return masked_image

    def separate_neighboring_objects(self, workspace, input_image, labeled_image, object_count):
        """Separate objects based on local maxima or distance transform

        workspace - get the image from here

        labeled_image - image labeled by scipy.ndimage.label

        object_count  - # of objects in image

        returns revised labeled_image, object count, maxima_suppression_size,
        LoG threshold and filter diameter
        """
        if self.advanced and (
            self.unclump_method == UN_NONE or self.watershed_method == WA_NONE
        ):
            return labeled_image, object_count, 7

        cpimage = input_image
        image = cpimage.pixel_data
        mask = cpimage.mask

        blurred_image = self.smooth_image(image, mask)
        if self.size_range[0] > 10 and (self.basic or self.low_res_maxima):
            image_resize_factor = 10.0 / float(self.size_range[0])
            if self.basic or self.automatic_suppression:
                maxima_suppression_size = 7
            else:
                maxima_suppression_size = (
                    self.maxima_suppression_size * image_resize_factor + 0.5
                )
            reported_maxima_suppression_size = (
                maxima_suppression_size / image_resize_factor
            )
        else:
            image_resize_factor = 1.0
            if self.basic or self.automatic_suppression:
                maxima_suppression_size = self.size_range[0] / 1.5
            else:
                maxima_suppression_size = self.maxima_suppression_size
            reported_maxima_suppression_size = maxima_suppression_size
        maxima_mask = centrosome.cpmorphology.strel_disk(
            max(1, maxima_suppression_size - 0.5)
        )
        distance_transformed_image = None
        if self.basic or self.unclump_method == UN_INTENSITY:
            # Remove dim maxima
            maxima_image = self.get_maxima(
                blurred_image, labeled_image, maxima_mask, image_resize_factor
            )
        elif self.unclump_method == UN_SHAPE:
            if self.fill_holes == FH_NEVER:
                # For shape, even if the user doesn't want to fill holes,
                # a point far away from the edge might be near a hole.
                # So we fill just for this part.
                foreground = (
                    centrosome.cpmorphology.fill_labeled_holes(labeled_image) > 0
                )
            else:
                foreground = labeled_image > 0
            distance_transformed_image = scipy.ndimage.distance_transform_edt(
                foreground
            )
            # randomize the distance slightly to get unique maxima
            numpy.random.seed(0)
            distance_transformed_image += numpy.random.uniform(
                0, 0.001, distance_transformed_image.shape
            )
            maxima_image = self.get_maxima(
                distance_transformed_image,
                labeled_image,
                maxima_mask,
                image_resize_factor,
            )
        else:
            raise ValueError(
                "Unsupported local maxima method: %s" % self.unclump_method
            )

        # Create the image for watershed
        if self.basic or self.watershed_method == WA_INTENSITY:
            # use the reverse of the image to get valleys at peaks
            watershed_image = 1 - image
        elif self.watershed_method == WA_SHAPE:
            if distance_transformed_image is None:
                distance_transformed_image = scipy.ndimage.distance_transform_edt(
                    labeled_image > 0
                )
            watershed_image = -distance_transformed_image
            watershed_image = watershed_image - numpy.min(watershed_image)
        elif self.watershed_method == WA_PROPAGATE:
            # No image used
            pass
        else:
            raise NotImplementedError(
                "Watershed method %s is not implemented" % self.watershed_method
            )
        #
        # Create a marker array where the unlabeled image has a label of
        # -(nobjects+1)
        # and every local maximum has a unique label which will become
        # the object's label. The labels are negative because that
        # makes the watershed algorithm use FIFO for the pixels which
        # yields fair boundaries when markers compete for pixels.
        #
        self.labeled_maxima, object_count = scipy.ndimage.label(
            maxima_image, numpy.ones((3, 3), bool)
        )
        if self.advanced and self.watershed_method == WA_PROPAGATE:
            watershed_boundaries, distance = centrosome.propagate.propagate(
                numpy.zeros(self.labeled_maxima.shape),
                self.labeled_maxima,
                labeled_image != 0,
                1.0,
            )
        else:
            markers_dtype = (
                numpy.int16
                if object_count < numpy.iinfo(numpy.int16).max
                else numpy.int32
            )
            markers = numpy.zeros(watershed_image.shape, markers_dtype)
            markers[self.labeled_maxima > 0] = -self.labeled_maxima[
                self.labeled_maxima > 0
            ]

            #
            # Some labels have only one maker in them, some have multiple and
            # will be split up.
            #

            watershed_boundaries = skimage.segmentation.watershed(
                connectivity=numpy.ones((3, 3), bool),
                image=watershed_image,
                markers=markers,
                mask=labeled_image != 0,
            )

            watershed_boundaries = -watershed_boundaries

        return watershed_boundaries, object_count, reported_maxima_suppression_size

    def get_maxima(self, image, labeled_image, maxima_mask, image_resize_factor):
        if image_resize_factor < 1.0:
            shape = numpy.array(image.shape) * image_resize_factor
            i_j = (
                numpy.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
                / image_resize_factor
            )
            resized_image = scipy.ndimage.map_coordinates(image, i_j)
            resized_labels = scipy.ndimage.map_coordinates(
                labeled_image, i_j, order=0
            ).astype(labeled_image.dtype)

        else:
            resized_image = image
            resized_labels = labeled_image
        #
        # find local maxima
        #
        if maxima_mask is not None:
            binary_maxima_image = centrosome.cpmorphology.is_local_maximum(
                resized_image, resized_labels, maxima_mask
            )
            binary_maxima_image[resized_image <= 0] = 0
        else:
            binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
        if image_resize_factor < 1.0:
            inverse_resize_factor = float(image.shape[0]) / float(
                binary_maxima_image.shape[0]
            )
            i_j = (
                numpy.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float)
                / inverse_resize_factor
            )
            binary_maxima_image = (
                scipy.ndimage.map_coordinates(binary_maxima_image.astype(float), i_j)
                > 0.5
            )
            assert binary_maxima_image.shape[0] == image.shape[0]
            assert binary_maxima_image.shape[1] == image.shape[1]

        # Erode blobs of touching maxima to a single point

        shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
        return shrunk_image

    def filter_on_size(self, labeled_image, object_count):
        """ Filter the labeled image based on the size range

        labeled_image - pixel image labels
        object_count - # of objects in the labeled image
        returns the labeled image, and the labeled image with the
        small objects removed
        """
        if self.exclude_size and object_count > 0:
            areas = scipy.ndimage.measurements.sum(
                numpy.ones(labeled_image.shape),
                labeled_image,
                numpy.array(list(range(0, object_count + 1)), dtype=numpy.int32),
            )
            areas = numpy.array(areas, dtype=int)
            min_allowed_area = (
                numpy.pi * (self.size_range[0] * self.size_range[0]) / 4
            )
            max_allowed_area = (
                numpy.pi * (self.size_range[1] * self.size_range[1]) / 4
            )
            # area_image has the area of the object at every pixel within the object
            area_image = areas[labeled_image]
            labeled_image[area_image < min_allowed_area] = 0
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
        else:
            small_removed_labels = labeled_image.copy()
        return labeled_image, small_removed_labels

    def filter_on_border(self, image, labeled_image):
        """Filter out objects touching the border

        In addition, if the image has a mask, filter out objects
        touching the border of the mask.
        """
        if self.exclude_border_objects:
            border_labels = list(labeled_image[0, :])
            border_labels.extend(labeled_image[:, 0])
            border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
            border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
            border_labels = numpy.array(border_labels)
            #
            # the following histogram has a value > 0 for any object
            # with a border pixel
            #
            histogram = scipy.sparse.coo_matrix(
                (
                    numpy.ones(border_labels.shape),
                    (border_labels, numpy.zeros(border_labels.shape)),
                ),
                shape=(numpy.max(labeled_image) + 1, 1),
            ).todense()
            histogram = numpy.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
            elif image.has_mask:
                # The assumption here is that, if nothing touches the border,
                # the mask is a large, elliptical mask that tells you where the
                # well is. That's the way the old Matlab code works and it's duplicated here
                #
                # The operation below gets the mask pixels that are on the border of the mask
                # The erosion turns all pixels touching an edge to zero. The not of this
                # is the border + formerly masked-out pixels.
                mask_border = numpy.logical_not(
                    scipy.ndimage.binary_erosion(image.mask)
                )
                mask_border = numpy.logical_and(mask_border, image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = scipy.sparse.coo_matrix(
                    (
                        numpy.ones(border_labels.shape),
                        (border_labels, numpy.zeros(border_labels.shape)),
                    ),
                    shape=(numpy.max(labeled_image) + 1, 1),
                ).todense()
                histogram = numpy.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    labeled_image[histogram_image > 0] = 0
        return labeled_image

    def display(self, workspace, figure):
        if self.show_window:
            """Display the image and labeling"""
            figure.set_subplots((2, 2))

            orig_axes = figure.subplot(0, 0)
            label_axes = figure.subplot(1, 0, sharexy=orig_axes)
            outlined_axes = figure.subplot(0, 1, sharexy=orig_axes)

            title = "Input image, cycle #%d" % (workspace.measurements.image_number,)
            image = workspace.display_data.image
            labeled_image = workspace.display_data.labeled_image
            size_excluded_labeled_image = workspace.display_data.size_excluded_labels
            border_excluded_labeled_image = (
                workspace.display_data.border_excluded_labels
            )

            ax = figure.subplot_imshow_grayscale(0, 0, image, title)
            figure.subplot_imshow_labels(
                1, 0, labeled_image, self.y_name, sharexy=ax
            )

            cplabels = [
                dict(name=self.y_name, labels=[labeled_image]),
                dict(
                    name="Objects filtered out by size",
                    labels=[size_excluded_labeled_image],
                ),
                dict(
                    name="Objects touching border",
                    labels=[border_excluded_labeled_image],
                ),
            ]
            if (
                self.unclump_method != UN_NONE
                and self.watershed_method != WA_NONE
                and self.want_plot_maxima
            ):
                # Generate static colormap for alpha overlay
                from matplotlib.colors import ListedColormap

                cmap = ListedColormap(self.maxima_color)
                if self.maxima_size > 1:
                    strel = skimage.morphology.disk(self.maxima_size - 1)
                    labels = skimage.morphology.dilation(self.labeled_maxima, selem=strel)
                else:
                    labels = self.labeled_maxima
                cplabels.append(
                    dict(
                        name="Detected maxima",
                        labels=[labels],
                        mode="alpha",
                        alpha_value=1,
                        alpha_colormap=cmap,
                    )
                )
            title = "%s outlines" % self.y_name
            figure.subplot_imshow_grayscale(
                0, 1, image, title, cplabels=cplabels, sharexy=ax
            )

            figure.subplot_table(
                1,
                1,
                [[x[1]] for x in workspace.display_data.statistics],
                row_labels=[x[0] for x in workspace.display_data.statistics],
            )

    def calc_smoothing_filter_size(self):
        """Return the size of the smoothing filter, calculating it if in automatic mode"""
        if self.automatic_smoothing:
            return 2.35 * self.size_range[0] / 3.5
        else:
            return self.smoothing_filter_size

    def is_object_identification_module(self):
        return True

    def _get_measurement_columns(self, pipeline, object_name=None):
        if object_name is None:
            object_name = self.y_name.value

        return [
            (object_name, M_LOCATION_CENTER_X, COLTYPE_FLOAT,),
            (object_name, M_LOCATION_CENTER_Y, COLTYPE_FLOAT,),
            (object_name, M_LOCATION_CENTER_Z, COLTYPE_FLOAT,),
            (object_name, M_NUMBER_OBJECT_NUMBER, COLTYPE_INTEGER,),
            (IMAGE, FF_COUNT % object_name, COLTYPE_INTEGER,),
        ]


    def get_measurement_columns(self, pipeline):
        columns = this._get_measurement_columns(pipeline)

        columns += self.threshold.get_measurement_columns(
            pipeline, object_name=self.y_name
        )

        return columns

    def _get_categories(self, pipeline, object_name):
        if object_name == IMAGE:
            return [C_COUNT]

        if object_name == self.y_name.value:
            return [
                C_LOCATION,
                C_NUMBER,
            ]

        return []

    def get_categories(self, pipeline, object_name):
        categories = self.threshold.get_categories(pipeline, object_name)

        categories += this._get_categories(
            pipeline, object_name
        )

        return categories

    def _get_measurements(self, pipeline, object_name, category):
        if object_name == IMAGE and category == C_COUNT:
            return [self.y_name.value]

        if object_name == self.y_name.value:
            if category == C_LOCATION:
                return [
                    FTR_CENTER_X,
                    FTR_CENTER_Y,
                    FTR_CENTER_Z,
                ]

            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]

        return []

    def get_measurements(self, pipeline, object_name, category):
        measurements = self.threshold.get_measurements(pipeline, object_name, category)

        measurements += this._get_measurements(
            pipeline, object_name, category
        )

        return measurements
    

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if measurement in self.threshold.get_measurements(
            pipeline, object_name, category
        ):
            return [self.y_name]

        return []
