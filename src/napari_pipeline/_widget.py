from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QRadioButton, QGroupBox, QProgressBar, QApplication, QScrollArea, QLineEdit, QFileDialog, QCheckBox, QHBoxLayout, QStyle, QSlider, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QSizePolicy, QAbstractScrollArea
from qtpy.QtGui import QIntValidator, QDoubleValidator
from qtpy import QtCore
from qtpy.QtCore import Qt, QEvent
import napari
import numpy as np
import os
import re
from napari_pipeline.CellProfilerImageWrapper import Image
from napari_pipeline.CellProfilerWorkspaceWrapper import Workspace
from napari_pipeline.modules.DynamicEnhanceOrSuppressFeatures import DynamicEnhanceOrSuppressFeatures
from napari_pipeline.modules.DynamicIdentifyPrimaryObjects import DynamicIdentifyPrimaryObjects
from napari_pipeline.modules.IdentifySecondaryObjects import IdentifySecondaryObjects
from multiprocessing import Pool
from functools import partial
import itertools
import json
from datetime import datetime
import csv


defaultSettingDictDynamicEnhanceOrSuppressFeatures = {'disabled': False, 'method': 'Enhance', 'object_size': 50, 'enhance_method': 'Speckles', 'hole_size': [1, 10], 'smoothing': 2.0, 'angle': 0.0, 'decay': 0.95, 'neurite_choice': 'Tubeness', 'speckle_accuracy': 'Slow', 'wants_rescale': False}
defaultSettingDictDynamicIdentifyPrimaryObjects = {'size_range': [1, 50], 'exclude_size': False, 'exclude_border_objects': False, 'unclump_method': 'Shape', 'watershed_method': 'Shape', 'smoothing_filter_size': 10, 'maxima_suppression_size': 15.0, 'low_res_maxima': False, 'fill_holes': 'After both thresholding and declumping', 'automatic_smoothing': True, 'automatic_suppression': True, 'limit_choice': 'Continue', 'maximum_object_count': 500, 'use_advanced': True, 'threshold.threshold_scope': 'Global', 'threshold.global_operation': 'Otsu', 'threshold.threshold_smoothing_scale': 1.3488, 'threshold.threshold_correction_factor': 0.8, 'threshold.threshold_range': [0.0, 1.0], 'threshold.manual_threshold': 1e-05, 'threshold.thresholding_measurement': 'None', 'threshold.two_class_otsu': 'Two classes', 'threshold.log_transform': False, 'threshold.assign_middle_to_foreground': 'Foreground', 'threshold.adaptive_window_size': 255, 'threshold.lower_outlier_fraction': 0.05, 'threshold.upper_outlier_fraction': 0.05, 'threshold.averaging_method': 'Mean', 'threshold.variance_method': 'Standard deviation', 'threshold.number_of_deviations': 2.0, 'threshold.local_operation': 'Otsu'}
defaultSettingDictObjectCount = {'object_count': 3}

def process_image(image_object, config):
    workspace = Workspace()

    # Unspeakable things had to be done to get these modules to work
    # Don't bother even looking at the code it's a mess
    
    filtered_image_object = None
    if image_object.channelNumber != 0:
        DynamicEnhanceOrSuppressFeaturesModule = DynamicEnhanceOrSuppressFeatures()
        filtered_image_object = DynamicEnhanceOrSuppressFeaturesModule.run(image_object, config.get("modules").get("DynamicEnhanceOrSuppressFeatures").get(image_object.markerName))
        DynamicIdentifyPrimaryObjectsModule = DynamicIdentifyPrimaryObjects()
        objects_instance = DynamicIdentifyPrimaryObjectsModule.run(filtered_image_object, config.get("modules").get("DynamicIdentifyPrimaryObjects").get(image_object.markerName), workspace)
    else:
        DynamicIdentifyPrimaryObjectsModule = DynamicIdentifyPrimaryObjects()
        IdentifySecondaryObjectsModule = IdentifySecondaryObjects()
        primary_objects_instance = DynamicIdentifyPrimaryObjectsModule.run(image_object, config.get("modules").get("DynamicIdentifyPrimaryObjects").get(image_object.markerName), workspace)
        # objects_instance = IdentifySecondaryObjectsModule.run(image_object, primary_objects_instance, workspace)
        objects_instance = primary_objects_instance
    
    return image_object, filtered_image_object, objects_instance

# Generates all possible intersections and subsections of the channels 
def intersect_and_subsect_lists(dict_of_lists):
    # Create a list to store the results
    results = []

    # Generate all possible combinations of the lists for intersections
    keys_list = list(dict_of_lists.keys())
    for r in range(2, len(dict_of_lists) + 1):
        for subset in itertools.combinations(keys_list, r):
            # Extract the keys and lists from the subset
            lists = [dict_of_lists[key] for key in subset]

            # Find the intersection of the lists in the subset
            intersection = lists[0]
            for current_list in lists[1:]:
                intersection = np.intersect1d(intersection, current_list)

            # Store the keys and intersection in the results
            results.append((subset, intersection))
    
    # Generate the power set for all subsections (excluding empty set and full set)
    power_set = list(itertools.chain.from_iterable(itertools.combinations(keys_list, r) for r in range(2, len(keys_list))))

    for subset in power_set:
    # Extract the keys and lists from the subset
        lists = [dict_of_lists[key] for key in subset]

        # Find the intersection of the lists in the subset
        intersection = lists[0]
        for current_list in lists[1:]:
            intersection = np.intersect1d(intersection, current_list)
        
        # Store the keys and intersection in the results
        results.append((tuple(subset), intersection))
        
        # Compute the difference with all other sets not in the subset
        other_keys = set(keys_list) - set(subset)
        for other_key in other_keys:
            difference = np.setdiff1d(intersection, dict_of_lists[other_key])
            neg_key_tuple = tuple(subset) + (-other_key,)
            results.append((neg_key_tuple, difference))
    
    # Handle single sets' differences with combinations of other sets
    for key in keys_list:
        single_list = dict_of_lists[key]
        other_keys = set(keys_list) - {key}
        for r in range(1, len(other_keys) + 1):
            for other_subset in itertools.combinations(other_keys, r):
                other_lists = [dict_of_lists[other_key] for other_key in other_subset]
                difference = single_list
                for other_list in other_lists:
                    difference = np.setdiff1d(difference, other_list)
                neg_key_tuple = (key,) + tuple(-k for k in other_subset)
                results.append((neg_key_tuple, difference))

    return results

# Dynamically organized csv row appending based on order of current row
# This function allows for arbitrary channel counts in the same CSV file
def update_csv(data_dict, csv_filename):
    fieldnames = []
    rows = []

    # Read existing data from CSV file
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass

    # Update fieldnames with new keys from data_dict
    if sum(key in fieldnames for key in data_dict.keys()) == len(fieldnames):
        # If the current data_dict contains all the keys of the current CSV file,
        # use it to determine column order
        fieldnames = list(data_dict.keys())
    else:
        for key in data_dict.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    # Update rows with data_dict values
    for row in rows:
        for key in data_dict.keys():
            if key not in row:
                row[key] = ''

    # Append new row with data_dict values
    new_row = {key: data_dict.get(key, '') for key in fieldnames}
    rows.append(new_row)

    # Write updated data to CSV file
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def create_image(baseMarkerLabels, labels):
    mask = np.isin(baseMarkerLabels, labels)
    baseMarkerLabelsWithChannelImage = np.zeros_like(baseMarkerLabels)
    baseMarkerLabelsWithChannelImage[mask] = baseMarkerLabels[mask]
    return baseMarkerLabelsWithChannelImage



class PipelineWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        main_layout = QVBoxLayout()

        self.layer_types = {"image": napari.layers.image.image.Image, "labels": napari.layers.labels.labels.Labels}

        run_button_label = QLabel("Click to run:")
        main_layout.addWidget(run_button_label)

        self.run_pipeline_button = QPushButton("Run")
        self.run_pipeline_button.clicked.connect(self.run_pipeline)
        main_layout.addWidget(self.run_pipeline_button)

        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)
        
        self.configFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

        self.config = self.readJSON(self.configFile)

        self.viewer.layers.events.inserted.connect(self.on_layers_changed)
        self.viewer.layers.events.removed.connect(self.on_layers_changed)

        self.addFolderSelector(main_layout, "Results directory:", self.config.get("results").get("path"), self.config.get("results").get("enabled"))
        self.addFileSelector(main_layout, "Import settings:")

        # self.markerGroupLayout = QVBoxLayout()
        # main_layout.addLayout(self.markerGroupLayout)
        
        # Create a QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a QWidget to hold the markerGroupLayout
        scroll_content = QWidget()
        self.markerGroupLayout = QVBoxLayout(scroll_content)

        # Set the layout for the scroll area's content
        scroll_area.setWidget(scroll_content)

        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)


    def readJSON(self, fileName):
        if os.path.exists(fileName):
            # Open and load the JSON file
            with open(fileName, 'r') as file:
                try:
                    # Parse JSON file
                    data = json.load(file)
                    return data
                except json.JSONDecodeError as e:
                    # Handle JSON decode error (invalid JSON)
                    print("An error occurred while parsing the JSON file:", e)
        else:
            print(f"The file {fileName} does not exist.")
        
        return {}
        
    def importConfig(self, fileName):
        newConfig = self.readJSON(fileName)
        if newConfig is not None:
            self.config = newConfig
            self.updateConfigFile()
            self.on_layers_changed()

    def updateConfigValue(self, keys, value):
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        self.updateConfigFile()

    def updateConfigFile(self):
        with open(self.configFile, 'w') as file:
            # Write JSON data
            json.dump(self.config, file, indent=4)

    def addFolderSelector(self, main_layout, labelText, folderPath="", enabled=True):
        folder_label = QLabel(labelText)
        main_layout.addWidget(folder_label)

        hBoxLayout = QHBoxLayout()

        folderPathLabel = QLineEdit(folderPath, self)
        folderPathLabel.textChanged.connect(lambda: self.updateConfigValue(["results", "path"], folderPathLabel.text()))
        folderPathLabel.setEnabled(bool(enabled))
        hBoxLayout.addWidget(folderPathLabel)

        openButton = QPushButton(self)
        icon = self.style().standardIcon(QStyle.SP_DirIcon)  # Using directory icon
        openButton.setIcon(icon)
        openButton.clicked.connect(lambda: self.showFolderDialog(folderPathLabel, folderPath))
        hBoxLayout.addWidget(openButton)

        # Adding checkbox for enabling/disabling
        checkbox = QCheckBox("Enabled", self)
        checkbox.setChecked(bool(enabled))
        checkbox.stateChanged.connect(lambda state: folderPathLabel.setEnabled(state == Qt.Checked))
        checkbox.stateChanged.connect(lambda state: self.updateConfigValue(["results", "enabled"], state == Qt.Checked))
        hBoxLayout.addWidget(checkbox)

        main_layout.addLayout(hBoxLayout)

    def showFolderDialog(self, folderPathLabel, folderPath):
        # Open a QFileDialog to select a folder
        options = QFileDialog.Options()

        # Get the selected folder path
        folderPath = QFileDialog.getExistingDirectory(self, "Select Folder", folderPath, options=options)

        if folderPath:
            # Set the selected folder path to the QLineEdit
            folderPathLabel.setText(folderPath)


    def addFileSelector(self, main_layout, labelText, fileName = ""):
        run_button_label = QLabel(labelText)
        main_layout.addWidget(run_button_label)

        hBoxLayout = QHBoxLayout()
        filePathLabel = QLineEdit(fileName, self)
        filePathLabel.textChanged.connect(lambda: self.importConfig(filePathLabel.text()))
        hBoxLayout.addWidget(filePathLabel)

        openButton = QPushButton(self)
        icon = self.style().standardIcon(QStyle.SP_FileIcon)
        openButton.setIcon(icon)
        openButton.clicked.connect(lambda: self.showFileDialog(filePathLabel, fileName))
        hBoxLayout.addWidget(openButton)
        main_layout.addLayout(hBoxLayout)



    def showFileDialog(self, filePathLabel, fileName):
        # Open a QFileDialog to select a file
        options = QFileDialog.Options()

        # Get the selected file path
        fileName, _ = QFileDialog.getOpenFileName(self, "Select File",fileName, "JSON Files (*.json);;All Files (*)", options=options)
        
        if fileName:
            # Set the selected file path to the QLineEdit
            filePathLabel.setText(fileName)

    def add_slider(self, group_layout, labelName, markerName, configValue):
        label = QLabel(labelName)
        group_layout.addWidget(label)

        slider = DoubleSlider()
        slider.setOrientation(Qt.Horizontal)
        
        value = self.config.get("modules").get(configValue[0]).get(configValue[1]).get(configValue[2])
        slider.setMinimum(0)
        slider.setMaximum(np.ceil(value * 2))
        slider.setValue(value)
        slider.configValue = configValue

        hBoxLayout = QHBoxLayout()
 
        min_value_input = EditableLabel()
        min_value_input.setValue(slider.minimum())
        hBoxLayout.addWidget(min_value_input)
        hBoxLayout.addStretch(1)

        max_value_input = EditableLabel()
        max_value_input.setValue(slider.maximum())
        hBoxLayout.addWidget(max_value_input)

        hBoxLayout2 = QHBoxLayout()
        hBoxLayout2.addStretch(1)
        value_input = EditableLabel()
        value_input.setValue(slider.value())
        hBoxLayout2.addWidget(value_input)

        slider.doubleValueChanged.connect(lambda: self.slider_value_changed(slider, value_input))

        input_field_list = [min_value_input, max_value_input, value_input]

        # Connect bounds fields to update methods
        min_value_input.valueChanged.connect(lambda: self.update_slider_values(slider, min_value_input, input_field_list, "min"))
        max_value_input.valueChanged.connect(lambda: self.update_slider_values(slider, max_value_input, input_field_list, "max"))
        value_input.valueChanged.connect(lambda: self.update_slider_values(slider, value_input, input_field_list, "value"))

        group_layout.addLayout(hBoxLayout)
        group_layout.addWidget(slider)
        group_layout.addLayout(hBoxLayout2)

    def add_slider_group(self, main_layout, markerName, channelNumber):
        group_box = QGroupBox(markerName.upper() + ":")
        group_layout = QVBoxLayout()

        group_box.setLayout(group_layout)
        group_layout.addSpacing(5)      


        if channelNumber != 0:      
            enableEnhanceOrSuppressFeaturesCheckbox = QCheckBox("Enabled")
            enableEnhanceOrSuppressFeaturesCheckbox.stateChanged.connect(lambda: self.update_checkbox_value(enableEnhanceOrSuppressFeaturesCheckbox, markerName))
            enableEnhanceOrSuppressFeaturesCheckbox.setChecked(not (self.config.get("modules").get("DynamicEnhanceOrSuppressFeatures").get(markerName).get("disabled")))
            group_layout.addWidget(enableEnhanceOrSuppressFeaturesCheckbox)
            group_layout.addSpacing(5)

            methodChoicesEnhanceOrSuppressFeaturesLabel = QLabel("Method:")
            group_layout.addWidget(methodChoicesEnhanceOrSuppressFeaturesLabel)

            methodOptions = ["Enhance", "Suppress"]
            methodChoicesEnhanceOrSuppressFeatures = CustomComboBox()
            methodChoicesEnhanceOrSuppressFeatures.addItems(methodOptions)
            methodChoicesEnhanceOrSuppressFeatures.currentIndexChanged.connect(lambda: self.update_combobox_value(methodChoicesEnhanceOrSuppressFeatures, markerName))
            methodChoicesEnhanceOrSuppressFeatures.setCurrentIndex(methodOptions.index(self.config.get("modules").get("DynamicEnhanceOrSuppressFeatures").get(markerName).get("method")))
            group_layout.addWidget(methodChoicesEnhanceOrSuppressFeatures)
            group_layout.addSpacing(5)

            self.add_slider(group_layout, "EnhanceOrSuppressFeatures Object Size:", markerName,
                            ["DynamicEnhanceOrSuppressFeatures", markerName.lower(), "object_size"])

        self.add_slider(group_layout, "DynamicIdentifyPrimaryObjects Correction Factor:", markerName,
                        ["DynamicIdentifyPrimaryObjects", markerName.lower(), "threshold.threshold_correction_factor"])

        if channelNumber != 0:
            group_layout.addSpacing(5)
            objectCountSpinBoxLayout = QHBoxLayout()

            label = QLabel("Minimum Object Count Required:")
            objectCountSpinBoxLayout.addWidget(label)

            objectCountSpinBox = CustomSpinBox()
            objectCountSpinBox.setMinimum(1)
            objectCountSpinBox.setMaximum(100)
            objectCountSpinBox.setValue(self.config.get("modules").get("ObjectCount").get(markerName).get("object_count"))

            objectCountSpinBox.valueChanged.connect(lambda: self.updateConfigValue(["modules", "ObjectCount", markerName, "object_count"], objectCountSpinBox.value()))
            objectCountSpinBoxLayout.addWidget(objectCountSpinBox)
            objectCountSpinBoxLayout.addStretch(0)
            group_layout.addLayout(objectCountSpinBoxLayout)
        
        main_layout.addWidget(group_box)

    def update_combobox_value(self, combobox, markerName):
        methodOptions = ["Enhance", "Suppress"]
        self.config["modules"]["DynamicEnhanceOrSuppressFeatures"][markerName]["method"] = methodOptions[combobox.currentIndex()]
        self.updateConfigFile()

    def update_checkbox_value(self, checkbox, markerName):
        self.config["modules"]["DynamicEnhanceOrSuppressFeatures"][markerName]["disabled"] = not checkbox.isChecked()
        self.updateConfigFile()

    def update_slider_values(self, slider, field, input_field_list, fieldType):
        new_value = field.value()

        if fieldType == "min":
            slider.setMinimum(new_value)
        
        elif fieldType == "max":
            slider.setMaximum(new_value)
            # input_field_list[0].setMaximum(new_value)
            # input_field_list[1].setMinimum(new_value)

        elif fieldType == "value":
            max_value = input_field_list[1].minimum()
            value_change = round(new_value - slider.value(), 2)

            if (new_value < slider.minimum() or new_value > slider.maximum()):
                # input_field_list[0].setMaximum(value_change + max_value)
                input_field_list[0].setValue(input_field_list[0].value() + value_change)
                # input_field_list[1].setMinimum(value_change + max_value)
                input_field_list[1].setValue(input_field_list[1].value() + value_change)
                slider.setMinimum(input_field_list[0].value())
                slider.setMaximum(input_field_list[1].value())
            slider.setValue(new_value)

    def slider_value_changed(self, slider, label):
        new_value = slider.value()
        label.setValue(new_value)
        self.config["modules"][slider.configValue[0]][slider.configValue[1]][slider.configValue[2]] = new_value
        self.updateConfigFile()


    def run_pipeline(self):
        self.update_image_layer_list()
        self.check_has_all_settings()

        self.image_objects_dict = {}
        self.objects_instance_dict = {}
        self.filtered_image_objects_dict = {}
        self.marker_dict = {}
        self.channel_dict = {}
        results_dict = {}

        results_dict["Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if (not self.check_has_all_settings()):
            self.run_pipeline_button.setEnabled(False)
            return

        for image in self.image_layers:
            self.image_objects_dict[image] = Image(image=self.dynamicGrayscale(image.data),
                                                    mask=self.binary_mask,
                                                    path_name = os.path.dirname(image.source.path),
                                                    file_name = os.path.basename(image.source.path)
                                                    )
            self.image_objects_dict[image].channelNumber = image.channelNumber                                       
            self.image_objects_dict[image].markerName = image.markerName                                       
            self.marker_dict[image.markerName] = image
            self.channel_dict[image.channelNumber] = image
            results_dict[f"Channel {image.channelNumber} marker name"] = image.markerName
            results_dict[f"Channel {image.channelNumber} path"] = image.source.path
        
        objects_instance_list = []
        print("Starting pool")
        pool = Pool(os.cpu_count())  # create a multiprocessing Pool
        results = pool.map(partial(process_image, config = self.config),
                           self.image_objects_dict.values())
        print("Finished pool")

        for result in results:
            self.filtered_image_objects_dict[self.marker_dict[result[0].markerName]] = result[1]
            self.objects_instance_dict[self.marker_dict[result[0].markerName]] = result[2]

            results_dict[f"Channel {self.marker_dict[result[0].markerName].channelNumber} detections"] = result[2].count

        if results is not None:
            statusString = "Objects detected:\n\t" + '\n\t'.join([f'{r[0].markerName}: {str(self.objects_instance_dict[self.marker_dict[r[0].markerName]].count)}' for r in results])
            statusString += "\nCells detected:"
        else:
            statusString = "No results found"

        print("Creating new layers")
        for layer in self.image_layers:
            if layer.markerName:
                filtered_image_objects_dict = self.filtered_image_objects_dict[layer]
                if self.layer_name_dict.get(layer.markerName) is None:
                    if filtered_image_objects_dict is not None:
                        new_filtered_image_layer = self.viewer.add_image(filtered_image_objects_dict.pixel_data,
                                                                         name = f"{layer.markerName} Filtered",
                                                                         visible = False)
                        # new_filtered_image_layer.export = True
                else:
                    if filtered_image_objects_dict is not None:
                        self.layer_name_dict[f"{layer.markerName} Filtered"].data = filtered_image_objects_dict.pixel_data
                        self.layer_name_dict[f"{layer.markerName} Filtered"].export = True

                
        for layer in self.image_layers:
            if layer.markerName:
                objects_instance = self.objects_instance_dict[layer]
                labels = objects_instance.get_labels()[0][0]

                if self.layer_name_dict.get(layer.markerName) is None:
                    new_objects_layer = self.viewer.add_labels(labels, name = layer.markerName)
                    # new_objects_layer.export = True
                    if (layer.channelNumber == 0):
                        # Make DAPI labels outline for greater visibility
                        new_objects_layer.contour = 3

                else:
                    self.layer_name_dict[layer.markerName].data = labels
                    # self.layer_name_dict[layer.markerName].export = True

        print("Find intersections with DAPI")
        baseMarkerLabelsWithChannelImages = {}
        if results is not None and self.channel_dict.get(0) is not None:
            baseMarkerLabels = self.objects_instance_dict[self.channel_dict[0]].get_labels()[0][0]
            baseMarkerLabelsUnique = np.unique(baseMarkerLabels)
            baseMarkerLabelsWithChannelDict = {}

            def get_indices_by_label(labels):
                """Efficiently gets indices for each label in a 2D array.

                Args:
                    labels: A 2D NumPy array of labels.

                Returns:
                    A dictionary where keys are unique labels and values are tuples of (row_indices, column_indices).
                """

                # Flatten the labels array for efficient indexing
                flat_labels = labels.ravel()

                # Get unique labels and their counts
                unique_labels, counts = np.unique(flat_labels, return_counts=True)

                # Pre-allocate arrays for indices to improve performance
                row_indices = np.empty(flat_labels.size, dtype=int)
                col_indices = np.empty(flat_labels.size, dtype=int)

                # Calculate row and column indices for all elements at once
                np.divmod(np.arange(flat_labels.size), labels.shape[1], out=(row_indices, col_indices))

                # Create the result dictionary
                positions_dict = {}
                for label, count in zip(unique_labels, counts):
                    # Use boolean indexing to extract indices for each label
                    label_mask = flat_labels == label
                    positions_dict[label] = (row_indices[label_mask], col_indices[label_mask])

                return positions_dict

            positions_dict = get_indices_by_label(baseMarkerLabels)

            for channel, image in self.channel_dict.items():
                if (channel == 0):
                    continue

                channelLabels = self.objects_instance_dict[self.channel_dict[channel]].get_labels()[0][0]

                labels_dict = {}

                for label in baseMarkerLabelsUnique:
                    # Get the labels of channelLabels for each baseMarker label
                    positions = positions_dict[label]
                    corresponding_labels = channelLabels[positions]
                    labels_dict[label] = corresponding_labels

                    # Add baseMarker labels with n (3) separate channel labels in it (essentially, n separate marker dots for a given marker)
                    if (channel not in baseMarkerLabelsWithChannelDict):
                            baseMarkerLabelsWithChannelDict[channel] = []
                    if (np.unique(labels_dict[label]).size > self.config.get("modules").get("ObjectCount").get(image.markerName).get("object_count") or
                        (np.sum(labels_dict[label] != 0) > 85) and image.markerName.lower() in ["fos", "cgrp", "vglut"] or
                        (np.sum(labels_dict[label] != 0) > 150) and image.markerName.lower() in ["tdt"]): #hack way to get around the issue of too perfect segmentations
                        baseMarkerLabelsWithChannelDict[channel].append(label)
            
            # Create intersections of each marker list
            baseMarkerLabelsWithChannelDict.update(intersect_and_subsect_lists(baseMarkerLabelsWithChannelDict))

            # Create new images for each set of baseMarker objects
            for channel, labels in baseMarkerLabelsWithChannelDict.items():
                baseMarkerLabelsWithChannelImages[channel] = create_image(baseMarkerLabels, labels)
        
        print("Adding combined labels")
        def custom_sort(item):
            item = item[0]
            # Sort first by absolute value
            abs_value = abs(item) if isinstance(item, int) else abs(item[0])
            # Then sort by the number of negative numbers in the tuples
            neg_count = 0
            number_count = 0
            if isinstance(item, tuple):
                neg_count = sum(1 for num in item if num < 0)
                number_count = len(item)
            return (neg_count, number_count, abs_value)

        # Use custom sorting order for marker dict
        for channels, labels in sorted(baseMarkerLabelsWithChannelImages.items(), key = custom_sort):
            baseMarkerName = self.channel_dict[0].markerName

            layerName = baseMarkerName + " with "
            statusName = "\n\t"
            keyName = "Cells found with "

            if isinstance(channels, tuple):
                channelsList = list(channels)
            else:
                channelsList = [channels]
            
            # Arrange channels 1, 2, 3,..., -1, -2, -3 (where negative refers to excluded subsections)
            channelsList = sorted(channelsList, key = lambda x: abs(x)-x)

            for i in range(len(channelsList)):
                channel_dict_key = abs(channelsList[i])
                
                layerVisibility = True
                # subsections between channels
                if (channelsList[i] < 0):
                    layerName += "no "
                    statusName += "no "
                    keyName += "no "
                    layerVisibility = False
                    

                layerName += self.channel_dict[channel_dict_key].markerName
                statusName += self.channel_dict[channel_dict_key].markerName
                keyName += f"channel {channel_dict_key}"

                if (i != len(channelsList) - 1):
                    if (len(channelsList) != 2):
                        layerName += ","
                        keyName += ","
                    layerName += " "
                    keyName += " "

                if (i == len(channelsList) - 2):
                    layerName += "and "
                    keyName += "and "
                
                if (i < len(channelsList) - 1):
                    statusName += " + "
            
            if self.layer_name_dict.get(layerName) is None:
                new_layer = self.viewer.add_labels(labels, name = layerName, visible = layerVisibility)
                new_layer.contour = 3
                # new_layer.export = True
            else:
                self.layer_name_dict[layerName].data = labels
                # self.layer_name_dict[layerName].export = True

            cellCount = len(baseMarkerLabelsWithChannelDict[channels])
            if np.isin(0, baseMarkerLabelsWithChannelDict[channels]):
                cellCount -= 1
            statusName += ": " + str(cellCount)
            results_dict[keyName] = cellCount
            statusString += statusName
        
        self.status_label.setText(statusString)

        resultsPath = self.config.get("results").get("path")
        if (self.config.get("results").get("enabled")) and os.path.isdir(resultsPath):
            update_csv(results_dict, os.path.join(resultsPath, "output.csv"))
            # for name, layer in self.layer_name_dict.items():
            #     if (getattr(layer, "export", None)):
            #       layer.save(os.path.join(resultsPath, name))


    # Dynamically determine which channel to use for grayscale and return said channel
    def dynamicGrayscale(self, image):

        # If there is an alpha channel, remove it
        if (image.shape[-1] > 3 and len(image.shape) > 2):
            image = image[:, :, :-1]

        # Sum over all dimensions except the last one to determine most used channel
        image_channel_sum = np.sum(image, axis=(tuple(range(len(image.shape)-1))))
        best_image_channel_index = np.argmax(image_channel_sum)
        
        # Returns best channel regardless of image rank to support 3D images
        return image[..., best_image_channel_index]


    def getMarkerName(self, layer):
        file_name = os.path.basename(layer.source.path)
        markerName = re.search("[^\W_]+(?=_[^\W_]+\.[^\W_]+$)", file_name).group().lower()
        return markerName

    def update_image_layer_list(self):
        self.image_layers = []
        self.layer_name_dict = {}
        self.binary_mask = None

        for layer in self.viewer.layers:
            self.layer_name_dict[layer.name] = layer
            if (layer.source.path):
                file_name = os.path.basename(layer.source.path)
                if (isinstance(layer, self.layer_types["image"]) and file_name is not None):
                    if (file_name.rfind("_ch") != -1):
                        markerName = self.getMarkerName(layer)
                        channelNumber = int(file_name[file_name.rfind("_ch") + 3 : file_name.rfind("_ch") + 5])
                        layer.markerName = markerName
                        layer.channelNumber = channelNumber
                        
                        # Put channels in order so from back to front, they render from lowest to highest channel number
                        self.image_layers.insert(channelNumber, layer)
                    elif (file_name.rfind("binary_mask") != -1 and self.getMarkerName(layer) == "binary"):
                        self.binary_mask = self.dynamicGrayscale(layer.data)

    def on_layers_changed(self):
        self.update_image_layer_list()
        missingMarkers = self.get_markers_if_missing()
        # self.run_pipeline_button.setEnabled(len(missingMarkers) == 0)
        self.run_pipeline_button.setEnabled(True)


        if (len(missingMarkers) > 0 and len(self.image_layers) > 0):
            self.status_label.setText("The following markers did not have existing values and were set to defaults: " + " ,".join([entry["marker"] for entry in missingMarkers]))

            for missingMarker in missingMarkers:
                # A missing root of the config will alway happen before other config issues
                if (not missingMarker.get("isLayer")):
                    if self.config is None:
                        self.config = {}
                    self.config.modules = {}
                    self.config.results = {}
                if (self.config["modules"].get("DynamicEnhanceOrSuppressFeatures") is None):
                    self.config["modules"]["DynamicEnhanceOrSuppressFeatures"] = {}
                if (self.config["modules"].get("DynamicIdentifyPrimaryObjects") is None):
                    self.config["modules"]["DynamicIdentifyPrimaryObjects"] = {}
                if (self.config["modules"]["ObjectCount"] is None):
                    self.config["modules"]["ObjectCount"] = {}

                if (missingMarker.get("channel") != 0):
                    self.config["modules"]["DynamicEnhanceOrSuppressFeatures"][missingMarker.get("marker")] = defaultSettingDictDynamicEnhanceOrSuppressFeatures
                    self.config["modules"]["ObjectCount"][missingMarker.get("marker")] = defaultSettingDictObjectCount
                self.config["modules"]["DynamicIdentifyPrimaryObjects"][missingMarker.get("marker")] = defaultSettingDictDynamicIdentifyPrimaryObjects
            
            self.updateConfigFile()

        self.clear_layout(self.markerGroupLayout)
        for layer in self.image_layers:
            self.add_slider_group(self.markerGroupLayout, layer.markerName, layer.channelNumber)
        self.markerGroupLayout.addStretch(0)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def check_has_all_settings(self):
        if (self.get_markers_if_missing()):
            return False
        return True

    def get_markers_if_missing(self):
        hasAllSettings = True
        missingMarkers = []
        for layer in self.image_layers:
            if (self.config is None or self.config.get("modules") is None):
                if layer.markerName not in missingMarkers:
                    missingMarkers.append({"marker": None, "channel": None, "isLayer": False})

            if layer.channelNumber != 0:
                if (self.config.get("modules").get("DynamicEnhanceOrSuppressFeatures") is None or self.config.get("modules").get("DynamicEnhanceOrSuppressFeatures").get(layer.markerName) is None):
                    if layer.markerName not in missingMarkers:
                        missingMarkers.append({"marker": layer.markerName, "channel": layer.channelNumber, "isLayer": True})

            if (self.config.get("modules").get("DynamicIdentifyPrimaryObjects") is None or self.config.get("modules").get("DynamicIdentifyPrimaryObjects").get(layer.markerName) is None):
                if layer.markerName not in missingMarkers:
                    missingMarkers.append({"marker": layer.markerName, "channel": layer.channelNumber, "isLayer": True})
        return missingMarkers

    def get_layer_names(self, type="all", exclude_hidden=True):
        layers = self.viewer.layers
        filtered_layers = []
        for layer in layers:
            if (type == "all" or isinstance(layer, self.layer_types[type])) and ((not exclude_hidden) or (exclude_hidden and "<hidden>" not in layer.name)):
                filtered_layers.append(layer.name)
        return filtered_layers

class CustomSpinBox(QSpinBox):
    def wheelEvent(self, event):
        # Remove scrolling functionality
        event.ignore()

class CustomComboBox(QComboBox):
    def wheelEvent(self, event):
        # Remove scrolling functionality
        event.ignore()

class DoubleSlider(QSlider):
    doubleValueChanged = QtCore.Signal(float)

    def __init__(self, decimals=2, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)
        self.installEventFilter(self)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def minimum(self):
        return float(super(DoubleSlider, self).minimum()) / self._multi

    def maximum(self):
        return float(super(DoubleSlider, self).maximum()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))
    
    def eventFilter(self, obj, event):
        if obj == self and event.type() == QEvent.Wheel:
            # Only change value if slider is selected
            if not self.hasFocus():
                return True

        return super().eventFilter(obj, event)


class KeyPressHandler(QtCore.QObject):
    """Custom key press handler"""
    escapePressed = QtCore.Signal(bool)
    returnPressed = QtCore.Signal(bool)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            event_key = event.key()
            if event_key == QtCore.Qt.Key_Escape:
                self.escapePressed.emit(True)
                return True
            if event_key == QtCore.Qt.Key_Return or event_key == QtCore.Qt.Key_Enter:
                self.returnPressed.emit(True)
                return True

        return QtCore.QObject.eventFilter(self, obj, event)

class EditableLabel(QWidget):
    """Editable label"""
    valueChanged = QtCore.Signal(str)
    def __init__(self, parent=None, **kwargs):
        QWidget.__init__(self, parent=parent)

        self.is_editable = kwargs.get("editable", True)
        self.keyPressHandler = KeyPressHandler(self)

        self.mainLayout = QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setObjectName("mainLayout")
        
        self.label = QLabel(self)
        self.label.setObjectName("label")
        self.mainLayout.addWidget(self.label)
        self.spinbox = QDoubleSpinBox(self)
        # self.spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons) 
        self.spinbox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.spinbox.setDecimals(2)
        self.spinbox.setRange(float('-inf'), float('inf'))
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setObjectName("spinbox")
        self.spinbox.setAlignment(self.label.alignment())
        self.mainLayout.addWidget(self.spinbox)
        self.mainLayout.addStretch(2)
        # hide the line edit initially
        self.spinbox.setHidden(True)

        # setup signals
        self.create_signals()

    def create_signals(self):
        # self.spinbox.installEventFilter(self.keyPressHandler)
        self.label.mousePressEvent = self.labelPressedEvent
        self.spinbox.editingFinished.connect(self.labelUpdatedAction)

        # give the spinbox both a `returnPressed` and `escapedPressed` action
        # self.keyPressHandler.escapePressed.connect(self.escapePressedAction)
        # self.keyPressHandler.returnPressed.connect(self.returnPressedAction)

    def text(self):
        """Standard QLabel text getter"""
        return self.label.text()

    def setValue(self, value):
        """Standard QLabel text setter"""
        self.label.blockSignals(True)
        self.label.setText(str(value))
        self.spinbox.setValue(value)
        self.label.blockSignals(False)

    def labelPressedEvent(self, event):
        """Set editable if the left mouse button is clicked"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setLabelEditableAction()

    def setLabelEditableAction(self):
        """Action to make the widget editable"""
        if not self.is_editable:
            return

        self.label.setHidden(True)
        self.label.blockSignals(True)
        self.spinbox.setHidden(False)
        # self.spinbox.setValue(float(self.label.text()))
        self.spinbox.blockSignals(False)
        self.spinbox.setFocus(QtCore.Qt.MouseFocusReason)
        self.spinbox.selectAll()

    def labelUpdatedAction(self):
        """Indicates the widget text has been updated"""
        new_value = round(self.spinbox.value(), 2)

        self.label.setText(str(new_value))
        self.valueChanged.emit(str(new_value))
        
        self.spinbox.setHidden(True)
        self.label.setHidden(False)
        self.spinbox.blockSignals(True)
        self.label.blockSignals(False)

    def returnPressedAction(self):
        """Return/enter event handler"""
        self.labelUpdatedAction()

    def escapePressedAction(self):
        """Escape event handler"""
        self.label.setHidden(False)
        self.spinbox.setHidden(True)
        self.spinbox.blockSignals(True)
        self.label.blockSignals(False)
    
    def minimum(self):
        return self.spinbox.minimum()

    def maximum(self):
        return self.spinbox.maximum()

    def value(self):
        return self.spinbox.value()