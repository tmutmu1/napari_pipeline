import centrosome.cpmorphology
import centrosome.propagate
import numpy
import scipy.ndimage
import skimage.segmentation

C_CHILDREN = "Children"
FF_CHILDREN_COUNT = "%s_%%s_Count" % C_CHILDREN
C_PARENT = "Parent"
FF_PARENT = "%s_%%s" % C_PARENT
FTR_CENTER_Z = "Center_Z"
FTR_CENTER_Y = "Center_Y"
FTR_CENTER_X = "Center_X"
C_LOCATION = "Location"
C_NUMBER = "Number"
FTR_OBJECT_NUMBER = "Object_Number"
C_COUNT = "Count"
FF_COUNT = "%s_%%s" % C_COUNT

from napari_pipeline.CellProfilerObjectsWrapper import Objects
from napari_pipeline.modules.Threshold import Threshold


# from cellprofiler.modules import _help, threshold

__doc__ = """\
IdentifySecondaryObjects
========================

**IdentifySecondaryObjects** identifies objects (e.g., cells)
using objects identified by another module (e.g., nuclei) as a starting
point.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also the other **Identify** modules.

What is a secondary object?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

{DEFINITION_OBJECT}

We define an
object as *secondary* when it can be found in an image by using another
cellular feature as a reference for guiding detection.

For densely-packed cells (such as those in a confluent monolayer),
determining the cell borders using a cell body stain can be quite
difficult since they often have irregular intensity patterns and are
lower-contrast with more diffuse staining. In addition, cells often
touch their neighbors making it harder to delineate the cell borders. It
is often easier to identify an organelle which is well separated
spatially (such as the nucleus) as an object first and then use that
object to guide the detection of the cell borders. See the
**IdentifyPrimaryObjects** module for details on how to identify a
primary object.

In order to identify the edges of secondary objects, this module
performs two tasks:

#. Finds the dividing lines between secondary objects that touch each
   other.
#. Finds the dividing lines between the secondary objects and the
   background of the image. In most cases, this is done by thresholding
   the image stained for the secondary objects.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

This module identifies secondary objects based on two types of input:

#. An *object* (e.g., nuclei) identified from a prior module. These are
   typically produced by an **IdentifyPrimaryObjects** module, but any
   object produced by another module may be selected for this purpose.
#. (*optional*) An *image* highlighting the image features defining the edges of the
   secondary objects (e.g., cell edges).
   This is typically a fluorescent stain for the cell body, membrane or
   cytoskeleton (e.g., phalloidin staining for actin). However, any
   image that produces these features can be used for this purpose. For
   example, an image processing module might be used to transform a
   brightfield image into one that captures the characteristics of a
   cell body fluorescent stain. This input is optional because you can
   instead define secondary objects as a fixed distance around each
   primary object.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

A set of secondary objects are produced by this module, which can be
used in downstream modules for measurement purposes or other operations.
Because each primary object is used as the starting point for producing
a corresponding secondary object, keep in mind the following points:

-  The primary object will always be completely contained within a
   secondary object. For example, nuclei are completely enclosed within
   cells identified by actin staining.
-  There will always be at most one secondary object for each primary
   object.

Once the module has finished processing, the module display window will
show the following panels;
note that these are just for display: you must use the **SaveImages**
module if you would like to save any of these images to the hard drive
(as well, the **OverlayOutlines** module or **ConvertObjectsToImage**
modules might be needed):

-  *Upper left:* The raw, original image.
-  *Upper right:* The identified objects shown as a color image where
   connected pixels that belong to the same object are assigned the same
   color (*label image*). Note that assigned colors
   are arbitrary; they are used simply to help you distinguish the
   various objects.
-  *Lower left:* The raw image overlaid with the colored outlines of the
   identified secondary objects. The objects are shown with the
   following colors:

   -  Magenta: Secondary objects
   -  Green: Primary objects

   If you need to change the color defaults, you can make adjustments in
   *File > Preferences*.
-  *Lower right:* A table showing some of the settings you chose,
   as well as those calculated by the module in order to produce
   the objects shown.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* The number of secondary objects identified.
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

-  *Parent:* The identity of the primary object associated with each
   secondary object.
-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the center
   of mass of the identified secondary objects.

 """#.format(
#     **{
#         "DEFINITION_OBJECT": _help.DEFINITION_OBJECT,
#         "HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS,
#     }
# )

M_PROPAGATION = "Propagation"
M_WATERSHED_G = "Watershed - Gradient"
M_WATERSHED_I = "Watershed - Image"
M_DISTANCE_N = "Distance - N"
M_DISTANCE_B = "Distance - B"

"""# of setting values other than thresholding ones"""
N_SETTING_VALUES = 10

"""Parent (seed) relationship of input objects to output objects"""
R_PARENT = "Parent"


class IdentifySecondaryObjects():
    module_name = "IdentifySecondaryObjects"

    variable_revision_number = 10

    category = "Object Processing"

    def __init__(self):
        self.threshold = Threshold()

        super(IdentifySecondaryObjects, self).__init__()

    def volumetric(self):
        return False

    def settings(self):
        settings = super(IdentifySecondaryObjects, self).settings()

        return (
            settings
            + [
                self.method,
                self.image_name,
                self.distance_to_dilate,
                self.regularization_factor,
                self.wants_discard_edge,
                self.wants_discard_primary,
                self.new_primary_objects_name,
                self.fill_holes,
            ]
            + [self.threshold_setting_version]
            + self.threshold.settings()[2:]
        )

    def visible_settings(self):
        visible_settings = [self.image_name]

        visible_settings += super(IdentifySecondaryObjects, self).visible_settings()

        visible_settings += [self.method]

        if self.method != M_DISTANCE_N:
            visible_settings += self.threshold.visible_settings()[2:]

        if self.method in (M_DISTANCE_B, M_DISTANCE_N):
            visible_settings += [self.distance_to_dilate]
        elif self.method == M_PROPAGATION:
            visible_settings += [self.regularization_factor]

        visible_settings += [self.fill_holes, self.wants_discard_edge]

        if self.wants_discard_edge:
            visible_settings += [self.wants_discard_primary]

            if self.wants_discard_primary:
                visible_settings += [self.new_primary_objects_name]

        return visible_settings

    def help_settings(self):
        help_settings = [self.x_name, self.y_name, self.method, self.image_name]

        help_settings += self.threshold.help_settings()[2:]

        help_settings += [
            self.distance_to_dilate,
            self.regularization_factor,
            self.fill_holes,
            self.wants_discard_edge,
            self.wants_discard_primary,
            self.new_primary_objects_name,
        ]

        return help_settings

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number < 9:
            raise NotImplementedError(
                "Automatic upgrade for this module is not supported in CellProfiler 3."
            )

        if variable_revision_number == 9:
            setting_values = (
                setting_values[:6] + setting_values[8:11] + setting_values[13:]
            )

            variable_revision_number = 10

        threshold_setting_values = setting_values[N_SETTING_VALUES:]

        threshold_settings_version = int(threshold_setting_values[0])

        if threshold_settings_version < 4:
            threshold_setting_values = self.threshold.upgrade_threshold_settings(
                threshold_setting_values
            )

            threshold_settings_version = 9

        (
            threshold_upgrade_settings,
            threshold_settings_version,
        ) = self.threshold.upgrade_settings(
            ["None", "None"] + threshold_setting_values[1:],
            threshold_settings_version,
            "Threshold",
        )

        threshold_upgrade_settings = [
            str(threshold_settings_version)
        ] + threshold_upgrade_settings[2:]

        setting_values = setting_values[:N_SETTING_VALUES] + threshold_upgrade_settings

        return setting_values, variable_revision_number
    
    def get_static_settings(self):
        self.distance_to_dilate = 10
        self.method = M_DISTANCE_N
        self.regularization_factor = 0.05
        self.wants_discard_edge = "No"
        self.fill_holes = "Yes"
        self.wants_discard_primary = False
        self.new_primary_objects_name = "FilteredNuclei"

        self.threshold.threshold_setting_version = self.threshold.variable_revision_number
        self.threshold.threshold_scope = "Global"
        self.threshold.global_operation = "Minimum Cross-Entropy"
        self.threshold.local_operation = "Minimum Cross-Entropy"
        self.threshold.threshold_smoothing_scale = 0
        self.threshold.threshold_correction_factor = 1
        self.threshold.threshold_range = (0, 1)
        self.threshold.manual_threshold = 0.0
        self.threshold.thresholding_measurement = lambda: "Image"
        self.threshold.two_class_otsu = None
        self.threshold.assign_middle_to_foreground = None
        self.threshold.upper_outlier_fraction = 0.05
        self.threshold.averaging_method = None
        self.threshold.variance_method = None
        self.threshold.number_of_deviations = 2
        self.threshold.adaptive_window_size = 50
        self.threshold.log_transform = False        

    def run(self, image, objects, workspace):
        self.get_static_settings()
        image_name = image.file_name
        workspace.display_data.statistics = []
        img = image.pixel_data
        mask = image.mask

        if img.shape != objects.shape:
            raise ValueError(
                "This module requires that the input image and object sets are the same size.\n"
                "The %s image and %s objects are not (%s vs %s).\n"
                "If they are paired correctly you may want to use the Resize, ResizeObjects or "
                "Crop module(s) to make them the same size."
                % (image_name, self.x_name.value, img.shape, objects.shape,)
            )
        global_threshold = None
        if self.method == M_DISTANCE_N:
            has_threshold = False
        else:
            thresholded_image, global_threshold, sigma = self._threshold_image(
                image, workspace
            )
            workspace.display_data.global_threshold = global_threshold
            workspace.display_data.threshold_sigma = sigma
            has_threshold = True

        #
        # Get the following labels:
        # * all edited labels
        # * labels touching the edge, including small removed
        #
        labels_in = objects.unedited_segmented.copy()
        labels_touching_edge = numpy.hstack(
            (labels_in[0, :], labels_in[-1, :], labels_in[:, 0], labels_in[:, -1])
        )
        labels_touching_edge = numpy.unique(labels_touching_edge)
        is_touching = numpy.zeros(numpy.max(labels_in) + 1, bool)
        is_touching[labels_touching_edge] = True
        is_touching = is_touching[labels_in]

        labels_in[(~is_touching) & (objects.segmented == 0)] = 0
        #
        # Stretch the input labels to match the image size. If there's no
        # label matrix, then there's no label in that area.
        #
        if tuple(labels_in.shape) != tuple(img.shape):
            tmp = numpy.zeros(img.shape, labels_in.dtype)
            i_max = min(img.shape[0], labels_in.shape[0])
            j_max = min(img.shape[1], labels_in.shape[1])
            tmp[:i_max, :j_max] = labels_in[:i_max, :j_max]
            labels_in = tmp

        if self.method in (M_DISTANCE_B, M_DISTANCE_N):
            if self.method == M_DISTANCE_N:
                distances, (i, j) = scipy.ndimage.distance_transform_edt(
                    labels_in == 0, return_indices=True
                )
                labels_out = numpy.zeros(labels_in.shape, int)
                dilate_mask = distances <= self.distance_to_dilate
                labels_out[dilate_mask] = labels_in[i[dilate_mask], j[dilate_mask]]
            else:
                labels_out, distances = centrosome.propagate.propagate(
                    img, labels_in, thresholded_image, 1.0
                )
                labels_out[distances > self.distance_to_dilate] = 0
                labels_out[labels_in > 0] = labels_in[labels_in > 0]
            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out
            #
            # Create the final output labels by removing labels in the
            # output matrix that are missing from the segmented image
            #
            segmented_labels = objects.segmented
            segmented_out = self.filter_labels(
                image, small_removed_segmented_out, objects, workspace
            )
        elif self.method == M_PROPAGATION:
            labels_out, distance = centrosome.propagate.propagate(
                img, labels_in, thresholded_image, self.regularization_factor
            )
            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(
                image, small_removed_segmented_out, objects, workspace
            )
        elif self.method == M_WATERSHED_G:
            #
            # First, apply the sobel filter to the image (both horizontal
            # and vertical). The filter measures gradient.
            #
            sobel_image = numpy.abs(scipy.ndimage.sobel(img))
            #
            # Combine the image mask and threshold to mask the watershed
            #
            watershed_mask = numpy.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = numpy.logical_and(watershed_mask, mask)

            #
            # Perform the first watershed
            #

            labels_out = skimage.segmentation.watershed(
                connectivity=numpy.ones((3, 3), bool),
                image=sobel_image,
                markers=labels_in,
                mask=watershed_mask,
            )

            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(
                image, small_removed_segmented_out, objects, workspace
            )
        elif self.method == M_WATERSHED_I:
            #
            # invert the image so that the maxima are filled first
            # and the cells compete over what's close to the threshold
            #
            inverted_img = 1 - img
            #
            # Same as above, but perform the watershed on the original image
            #
            watershed_mask = numpy.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = numpy.logical_and(watershed_mask, mask)
            #
            # Perform the watershed
            #

            labels_out = skimage.segmentation.watershed(
                connectivity=numpy.ones((3, 3), bool),
                image=inverted_img,
                markers=labels_in,
                mask=watershed_mask,
            )

            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out
            segmented_out = self.filter_labels(
                image, small_removed_segmented_out, objects, workspace
            )

        if self.wants_discard_edge:
            lookup = scipy.ndimage.maximum(
                segmented_out,
                objects.segmented,
                list(range(numpy.max(objects.segmented) + 1)),
            )
            lookup = centrosome.cpmorphology.fixup_scipy_ndimage_result(lookup)
            lookup[0] = 0
            lookup[lookup != 0] = numpy.arange(numpy.sum(lookup != 0)) + 1
            segmented_labels = lookup[objects.segmented]
            segmented_out = lookup[segmented_out]

        
            if self.wants_discard_primary:
                #
                # Make a new primary object
                #
                new_objects = Objects()
                new_objects.segmented = segmented_labels
                if objects.has_unedited_segmented:
                    new_objects.unedited_segmented = objects.unedited_segmented
                if objects.has_small_removed_segmented:
                    new_objects.small_removed_segmented = objects.small_removed_segmented
                new_objects.parent_image = objects.parent_image

        #
        # Add the objects to the object set
        #
        objects_out = Objects()
        objects_out.unedited_segmented = small_removed_segmented_out
        objects_out.small_removed_segmented = small_removed_segmented_out
        objects_out.segmented = segmented_out
        objects_out.parent_image = image
        # objname = self.y_name.value
        # workspace.object_set.add_objects(objects_out, objname)
        object_count = numpy.max(segmented_out)
        return objects_out
        #
        # Add measurements
        #
        measurements = workspace.measurements
        super(IdentifySecondaryObjects, self).add_measurements(workspace)
        #
        # Relate the secondary objects to the primary ones and record
        # the relationship.
        #
        children_per_parent, parents_of_children = objects.relate_children(objects_out)
        measurements.add_measurement(
            self.x_name.value, FF_CHILDREN_COUNT % objname, children_per_parent,
        )
        measurements.add_measurement(
            objname, FF_PARENT % self.x_name.value, parents_of_children,
        )
        image_numbers = (
            numpy.ones(len(parents_of_children), int) * measurements.image_set_number
        )
        mask = parents_of_children > 0
        measurements.add_relate_measurement(
            self.module_num,
            R_PARENT,
            self.x_name.value,
            self.y_name.value,
            image_numbers[mask],
            parents_of_children[mask],
            image_numbers[mask],
            numpy.arange(1, len(parents_of_children) + 1)[mask],
        )
        #
        # If primary objects were created, add them
        #
        if self.wants_discard_edge and self.wants_discard_primary:
            workspace.object_set.add_objects(
                new_objects, self.new_primary_objects_name
            )
            super(IdentifySecondaryObjects, self).add_measurements(
                workspace,
                input_object_name=self.x_name.value,
                output_object_name=self.new_primary_objects_name,
            )

            children_per_parent, parents_of_children = new_objects.relate_children(
                objects_out
            )

            measurements.add_measurement(
                self.new_primary_objects_name,
                FF_CHILDREN_COUNT % objname,
                children_per_parent,
            )

            measurements.add_measurement(
                objname,
                FF_PARENT % self.new_primary_objects_name,
                parents_of_children,
            )

        if self.show_window:
            object_area = numpy.sum(segmented_out > 0)
            workspace.display_data.object_pct = (
                100 * object_area / numpy.product(segmented_out.shape)
            )
            workspace.display_data.img = img
            workspace.display_data.segmented_out = segmented_out
            workspace.display_data.primary_labels = objects.segmented
            workspace.display_data.global_threshold = global_threshold
            workspace.display_data.object_count = object_count

    def _threshold_image(self, image, workspace, automatic=False):

        final_threshold, orig_threshold, guide_threshold, binary_image, sigma = self.threshold.get_threshold(
            image, workspace, automatic
        )

        self.threshold.add_threshold_measurements(
            self.y_name.value,
            workspace.measurements,
            final_threshold,
            orig_threshold,
            guide_threshold,
        )

        self.threshold.add_fg_bg_measurements(
            self.y_name.value, workspace.measurements, image, binary_image
        )

        return binary_image, numpy.mean(numpy.atleast_1d(final_threshold)), sigma

    def display(self, workspace, figure):
        object_pct = workspace.display_data.object_pct
        img = workspace.display_data.img
        primary_labels = workspace.display_data.primary_labels
        segmented_out = workspace.display_data.segmented_out
        global_threshold = workspace.display_data.global_threshold
        object_count = workspace.display_data.object_count
        statistics = workspace.display_data.statistics

        if global_threshold is not None:
            statistics.append(["Threshold", "%0.3g" % global_threshold])

        if object_count > 0:
            areas = scipy.ndimage.sum(
                numpy.ones(segmented_out.shape),
                segmented_out,
                numpy.arange(1, object_count + 1),
            )
            areas.sort()
            low_diameter = numpy.sqrt(float(areas[object_count // 10]) / numpy.pi) * 2
            median_diameter = numpy.sqrt(float(areas[object_count // 2]) / numpy.pi) * 2
            high_diameter = (
                numpy.sqrt(float(areas[object_count * 9 // 10]) / numpy.pi) * 2
            )
            statistics.append(["10th pctile diameter", "%.1f pixels" % low_diameter])
            statistics.append(["Median diameter", "%.1f pixels" % median_diameter])
            statistics.append(["90th pctile diameter", "%.1f pixels" % high_diameter])
            if self.method != M_DISTANCE_N:
                statistics.append(
                    [
                        "Thresholding filter size",
                        "%.1f" % workspace.display_data.threshold_sigma,
                    ]
                )
            statistics.append(["Area covered by objects", "%.1f %%" % object_pct])
        workspace.display_data.statistics = statistics

        figure.set_subplots((2, 2))
        title = "Input image, cycle #%d" % workspace.measurements.image_number
        figure.subplot_imshow_grayscale(0, 0, img, title)
        figure.subplot_imshow_labels(
            1,
            0,
            segmented_out,
            "%s objects" % self.y_name.value,
            sharexy=figure.subplot(0, 0),
        )

        cplabels = [
            dict(name=self.x_name.value, labels=[primary_labels]),
            dict(name=self.y_name.value, labels=[segmented_out]),
        ]
        title = "%s and %s outlines" % (self.x_name.value, self.y_name.value)
        figure.subplot_imshow_grayscale(
            0, 1, img, title=title, cplabels=cplabels, sharexy=figure.subplot(0, 0)
        )
        figure.subplot_table(
            1,
            1,
            [[x[1]] for x in workspace.display_data.statistics],
            row_labels=[x[0] for x in workspace.display_data.statistics],
        )

    def filter_labels(self, image, labels_out, objects, workspace):
        """Filter labels out of the output

        Filter labels that are not in the segmented input labels. Optionally
        filter labels that are touching the edge.

        labels_out - the unfiltered output labels
        objects    - the objects thing, containing both segmented and
                     small_removed labels
        """
        segmented_labels = objects.segmented
        max_out = numpy.max(labels_out)
        if max_out > 0:
            segmented_labels, m1 = size_similarly(labels_out, segmented_labels)
            segmented_labels[~m1] = 0
            lookup = scipy.ndimage.maximum(
                segmented_labels, labels_out, list(range(max_out + 1))
            )
            lookup = numpy.array(lookup, int)
            lookup[0] = 0
            segmented_labels_out = lookup[labels_out]
        else:
            segmented_labels_out = labels_out.copy()
        if self.wants_discard_edge:
            if image.has_mask:
                mask_border = image.mask & ~scipy.ndimage.binary_erosion(image.mask)
                edge_labels = segmented_labels_out[mask_border]
            else:
                edge_labels = numpy.hstack(
                    (
                        segmented_labels_out[0, :],
                        segmented_labels_out[-1, :],
                        segmented_labels_out[:, 0],
                        segmented_labels_out[:, -1],
                    )
                )
            edge_labels = numpy.unique(edge_labels)
            #
            # Make a lookup table that translates edge labels to zero
            # but translates everything else to itself
            #
            lookup = numpy.arange(max(max_out, numpy.max(segmented_labels)) + 1)
            lookup[edge_labels] = 0
            #
            # Run the segmented labels through this to filter out edge
            # labels
            segmented_labels_out = lookup[segmented_labels_out]

        return segmented_labels_out

    def is_object_identification_module(self):
        return True

    def get_measurement_columns(self, pipeline):
        if self.wants_discard_edge and self.wants_discard_primary:
            columns = super(IdentifySecondaryObjects, self).get_measurement_columns(
                pipeline,
                additional_objects=[
                    (self.x_name.value, self.new_primary_objects_name)
                ],
            )

            columns += [
                (
                    self.new_primary_objects_name,
                    FF_CHILDREN_COUNT % self.y_name.value,
                    "integer",
                ),
                (
                    self.y_name.value,
                    FF_PARENT % self.new_primary_objects_name,
                    "integer",
                ),
            ]
        else:
            columns = super(IdentifySecondaryObjects, self).get_measurement_columns(
                pipeline
            )

        if self.method != M_DISTANCE_N:
            columns += self.threshold.get_measurement_columns(
                pipeline, object_name=self.y_name.value
            )

        return columns

    def get_categories(self, pipeline, object_name):
        categories = super(IdentifySecondaryObjects, self).get_categories(
            pipeline, object_name
        )

        if self.method != M_DISTANCE_N:
            categories += self.threshold.get_categories(pipeline, object_name)

        if self.wants_discard_edge and self.wants_discard_primary:
            if object_name == self.new_primary_objects_name:
                # new_primary_objects_name objects has the same categories as y_name objects
                categories += super(IdentifySecondaryObjects, self).get_categories(
                    pipeline, self.y_name.value
                )

                categories += [C_CHILDREN]

        return categories

    def get_measurements(self, pipeline, object_name, category):
        measurements = super(IdentifySecondaryObjects, self).get_measurements(
            pipeline, object_name, category
        )

        if self.method != M_DISTANCE_N:
            measurements += self.threshold.get_measurements(
                pipeline, object_name, category
            )

        if self.wants_discard_edge and self.wants_discard_primary:
            if object_name == "Image" and category == C_COUNT:
                measurements += [self.new_primary_objects_name]

            if object_name == self.y_name.value and category == C_PARENT:
                measurements += [self.new_primary_objects_name]

            if object_name == self.new_primary_objects_name:
                if category == C_LOCATION:
                    measurements += [
                        FTR_CENTER_X,
                        FTR_CENTER_Y,
                        FTR_CENTER_Z,
                    ]

                if category == C_NUMBER:
                    measurements += [FTR_OBJECT_NUMBER]

                if category == C_PARENT:
                    measurements += [self.x_name.value]

            if category == C_CHILDREN:
                if object_name == self.x_name.value:
                    measurements += ["%s_Count" % self.new_primary_objects_name]

                if object_name == self.new_primary_objects_name:
                    measurements += ["%s_Count" % self.y_name.value]

        return measurements

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        threshold_measurements = self.threshold.get_measurements(
            pipeline, object_name, category
        )

        if self.method != M_DISTANCE_N and measurement in threshold_measurements:
            return [self.y_name.value]

        return []

def size_similarly(labels, secondary):
    """Size the secondary matrix similarly to the labels matrix

    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).

    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    """
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, numpy.ones(secondary.shape, bool)
    if labels.shape[0] <= secondary.shape[0] and labels.shape[1] <= secondary.shape[1]:
        if secondary.ndim == 2:
            return (
                secondary[: labels.shape[0], : labels.shape[1]],
                numpy.ones(labels.shape, bool),
            )
        else:
            return (
                secondary[: labels.shape[0], : labels.shape[1], :],
                numpy.ones(labels.shape, bool),
            )

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = numpy.zeros(
        list(labels.shape) + list(secondary.shape[2:]), secondary.dtype
    )
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = numpy.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask