"""
EnhanceOrSuppressFeatures
=========================

**EnhanceOrSuppressFeatures** enhances or suppresses certain image
features (such as speckles, ring shapes, and neurites), which can
improve subsequent identification of objects.

This module enhances or suppresses the intensity of certain pixels
relative to the rest of the image, by applying image processing filters
to the image. It produces a grayscale image in which objects can be
identified using an **Identify** module.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============
"""

import centrosome.filter
import numpy
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.transform
from napari_pipeline.CellProfilerImageWrapper import Image

# import cellprofiler_core.module

import os
import json
import textwrap
import re

ENHANCE = "Enhance"
SUPPRESS = "Suppress"

E_SPECKLES = "Speckles"
E_NEURITES = "Neurites"
E_DARK_HOLES = "Dark holes"
E_CIRCLES = "Circles"
E_TEXTURE = "Texture"
E_DIC = "DIC"

S_FAST = "Fast"
S_SLOW = "Slow"

N_GRADIENT = "Line structures"
N_TUBENESS = "Tubeness"


class DynamicEnhanceOrSuppressFeatures():
    module_name = "DynamicEnhanceOrSuppressFeatures"

    variable_revision_number = 7 

    def __init__(self):
        self.disabled = False

    def settings(self):
        __settings__ = super(DynamicEnhanceOrSuppressFeatures, self).settings()
        return __settings__ + [
            self.method,
            self.object_size,
            self.enhance_method,
            self.hole_size,
            self.smoothing,
            self.angle,
            self.decay,
            self.neurite_choice,
            self.speckle_accuracy,
            self.wants_rescale,
            self.settings_text_file,
            self.settings_text_directory,
            self.get_settings_button,
        ]

    def visible_settings(self):
        
        __settings__ = [self.settings_text_file, self.get_settings_button]

        __settings__ += super(DynamicEnhanceOrSuppressFeatures, self).visible_settings()
        __settings__ += [self.method]
        if self.method == ENHANCE:
            __settings__ += [self.enhance_method]
            self.object_size.min_value = 2
            if self.enhance_method == E_DARK_HOLES:
                __settings__ += [self.hole_size]
            elif self.enhance_method == E_TEXTURE:
                __settings__ += [self.smoothing]
            elif self.enhance_method == E_DIC:
                __settings__ += [self.smoothing, self.angle, self.decay]
            elif self.enhance_method == E_NEURITES:
                __settings__ += [self.neurite_choice]
                if self.neurite_choice == N_GRADIENT:
                    __settings__ += [self.object_size]
                else:
                    __settings__ += [self.smoothing]
                __settings__ += [self.wants_rescale]
            elif self.enhance_method == E_SPECKLES:
                __settings__ += [self.object_size, self.speckle_accuracy]
                self.object_size.min_value = 3
            else:
                __settings__ += [self.object_size]
        else:
            __settings__ += [self.object_size]
        return __settings__

    # Get dynamic settings based on input image
    def get_dynamic_settings(self, settingsDict):
        for key, value in settingsDict.items():
            setattr(self, key, value)
            # markerName = re.search("[^\W_]+(?=_[^\W_]+\.[^\W_]+$)", input_image.file_name)
            # if (markerName is not None):
            #     markerSettings = settingsDict.get(markerName.group().lower())
            #     if (markerSettings is not None):
            #         self.disabled = markerSettings.get("disabled")
            #         self.method = markerSettings.get("method")
            #         self.object_size = markerSettings.get("object_size")
            #         self.enhance_method = markerSettings.get("enhance_method")
            #         self.hole_size = markerSettings.get("hole_size")
            #         self.smoothing = markerSettings.get("smoothing")
            #         self.angle = markerSettings.get("angle")
            #         self.decay = markerSettings.get("decay")
            #         self.neurite_choice = markerSettings.get("neurite_choice")
            #         self.speckle_accuracy = markerSettings.get("speckle_accuracy")
            #         self.wants_rescale = markerSettings.get("wants_rescale")
            #     else:
            #         raise ValueError("Error: Marker \"" + markerName.group() + "\" does not have an entry in the settings file")

    def run(self, image, settingsDict):
        
        self.get_dynamic_settings(settingsDict)

        if (self.disabled is True):
            result = image.pixel_data
            result_image = Image(result, parent_image=image, dimensions=image.dimensions)

        else:
            radius = self.object_size / 2

            if self.method == ENHANCE:
                if self.enhance_method == E_SPECKLES:
                    result = self.enhance_speckles(
                        image, radius, self.speckle_accuracy
                    )
                elif self.enhance_method == E_NEURITES:
                    result = self.enhance_neurites(image, radius, self.neurite_choice)
                    if self.wants_rescale:
                        result = skimage.exposure.rescale_intensity(result)
                elif self.enhance_method == E_DARK_HOLES:
                    min_radius = max(1, int(self.hole_size.min / 2))

                    max_radius = int((self.hole_size.max + 1) / 2)

                    result = self.enhance_dark_holes(image, min_radius, max_radius)
                elif self.enhance_method == E_CIRCLES:
                    result = self.enhance_circles(image, radius)
                elif self.enhance_method == E_TEXTURE:
                    result = self.enhance_texture(image, self.smoothing)
                elif self.enhance_method == E_DIC:
                    result = self.enhance_dic(
                        image, self.angle, self.decay, self.smoothing
                    )
                else:
                    raise NotImplementedError(
                        "Unimplemented enhance method: %s" % self.enhance_method
                    )
            elif self.method == SUPPRESS:
                result = self.suppress(image, radius)
            else:
                raise ValueError("Unknown filtering method: %s" % self.method)

            result_image = Image(result, parent_image=image, dimensions=image.dimensions)

        return result_image


    def __mask(self, pixel_data, mask):
        data = numpy.zeros_like(pixel_data)

        data[mask] = pixel_data[mask]

        return data

    def __unmask(self, data, pixel_data, mask):
        data[~mask] = pixel_data[~mask]

        return data

    def __structuring_element(self, radius, volumetric):
        if volumetric:
            return skimage.morphology.ball(radius)

        return skimage.morphology.disk(radius)

    def enhance_speckles(self, image, radius, accuracy):
        data = self.__mask(image.pixel_data, image.mask)

        footprint = self.__structuring_element(radius, image.volumetric)

        if accuracy == "Slow" or radius <= 3:
            result = skimage.morphology.white_tophat(data, footprint=footprint)
        else:
            #
            # white_tophat = img - opening
            #              = img - dilate(erode)
            #              = img - maximum_filter(minimum_filter)
            minimum = scipy.ndimage.filters.minimum_filter(data, footprint=selem)

            maximum = scipy.ndimage.filters.maximum_filter(minimum, footprint=selem)

            result = data - maximum

        return self.__unmask(result, image.pixel_data, image.mask)

    def enhance_neurites(self, image, radius, method):
        data = self.__mask(image.pixel_data, image.mask)

        if method == N_GRADIENT:
            # desired effect = img + white_tophat - black_tophat
            selem = self.__structuring_element(radius, image.volumetric)

            white = skimage.morphology.white_tophat(data, selem=selem)

            black = skimage.morphology.black_tophat(data, selem=selem)

            result = data + white - black

            result[result > 1] = 1

            result[result < 0] = 0
        else:
            sigma = self.smoothing

            smoothed = scipy.ndimage.gaussian_filter(
                data, numpy.divide(sigma, image.spacing)
            )

            if image.volumetric:
                result = numpy.zeros_like(smoothed)

                for index, plane in enumerate(smoothed):
                    hessian = centrosome.filter.hessian(
                        plane, return_hessian=False, return_eigenvectors=False
                    )

                    result[index] = (
                        -hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2)
                    )
            else:
                hessian = centrosome.filter.hessian(
                    smoothed, return_hessian=False, return_eigenvectors=False
                )

                #
                # The positive values are darker pixels with lighter
                # neighbors. The original ImageJ code scales the result
                # by sigma squared - I have a feeling this might be
                # a first-order correction for e**(-2*sigma), possibly
                # because the hessian is taken from one pixel away
                # and the gradient is less as sigma gets larger.
                result = -hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2)

        return self.__unmask(result, image.pixel_data, image.mask)

    def enhance_circles(self, image, radius):
        data = self.__mask(image.pixel_data, image.mask)

        if image.volumetric:
            result = numpy.zeros_like(data)

            for index, plane in enumerate(data):
                result[index] = skimage.transform.hough_circle(plane, radius)[0]
        else:
            result = skimage.transform.hough_circle(data, radius)[0]

        return self.__unmask(result, image.pixel_data, image.mask)

    def enhance_texture(self, image, sigma):
        mask = image.mask

        data = self.__mask(image.pixel_data, mask)

        gmask = skimage.filters.gaussian(
            mask.astype(float), sigma, mode="constant", multichannel=False
        )

        img_mean = (
            skimage.filters.gaussian(data, sigma, mode="constant", multichannel=False)
            / gmask
        )

        img_squared = (
            skimage.filters.gaussian(
                data ** 2, sigma, mode="constant", multichannel=False
            )
            / gmask
        )

        result = img_squared - img_mean ** 2

        return self.__unmask(result, image.pixel_data, mask)

    def enhance_dark_holes(self, image, min_radius, max_radius):
        pixel_data = image.pixel_data

        mask = image.mask if image.has_mask else None

        se = self.__structuring_element(1, image.volumetric)

        inverted_image = pixel_data.max() - pixel_data

        previous_reconstructed_image = inverted_image

        eroded_image = inverted_image

        smoothed_image = numpy.zeros(pixel_data.shape)

        for i in range(max_radius + 1):
            eroded_image = skimage.morphology.erosion(eroded_image, se)

            if mask is not None:
                eroded_image *= mask

            reconstructed_image = skimage.morphology.reconstruction(
                eroded_image, inverted_image, "dilation", se
            )

            output_image = previous_reconstructed_image - reconstructed_image

            if i >= min_radius:
                smoothed_image = numpy.maximum(smoothed_image, output_image)

            previous_reconstructed_image = reconstructed_image

        return smoothed_image

    def enhance_dic(self, image, angle, decay, smoothing):
        pixel_data = image.pixel_data

        if image.volumetric:
            result = numpy.zeros_like(pixel_data)

            for index, plane in enumerate(pixel_data):
                result[index] = centrosome.filter.line_integration(
                    plane, angle, decay, smoothing
                )

            return result

        if smoothing == 0:
            smoothing = numpy.finfo(float).eps

        return centrosome.filter.line_integration(pixel_data, angle, decay, smoothing)

    def suppress(self, image, radius):
        data = self.__mask(image.pixel_data, image.mask)

        selem = self.__structuring_element(radius, image.volumetric)

        result = skimage.morphology.opening(data, selem)

        return self.__unmask(result, image.pixel_data, image.mask)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust setting values if they came from a previous revision

        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        """
        if variable_revision_number == 1:
            #
            # V1 -> V2, added enhance method and hole size
            #
            setting_values = setting_values + [E_SPECKLES, "1,10"]
            variable_revision_number = 2
        if variable_revision_number == 2:
            #
            # V2 -> V3, added texture and DIC
            #
            setting_values = setting_values + ["2.0", "0", ".95"]
            variable_revision_number = 3
        if variable_revision_number == 3:
            setting_values = setting_values + [N_GRADIENT]
            variable_revision_number = 4
        if variable_revision_number == 4:
            setting_values = setting_values + ["Slow / circular"]
            variable_revision_number = 5

        if variable_revision_number == 5:
            if setting_values[-1] == "Slow / circular":
                setting_values[-1] = "Slow"
            else:
                setting_values[-1] = "Fast"

            variable_revision_number = 6

        if variable_revision_number == 6:
            # Add neurite rescaling option
            setting_values.append("Yes")
            variable_revision_number = 7

        return setting_values, variable_revision_number


EnhanceOrSuppressSpeckles = DynamicEnhanceOrSuppressFeatures
