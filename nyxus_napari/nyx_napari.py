from typing import Union
from qtpy.QtWidgets import QWidget, QScrollArea, QTableWidget, QLineEdit, QVBoxLayout, QHBoxLayout, QTableWidgetItem, QLabel, QAbstractSlider, QSlider
from qtpy.QtCore import Qt, QSize
from qtpy import QtCore, QtGui, QtWidgets, uic
import napari
from napari.layers import Image
from napari.utils.notifications import show_info
from magicgui import magic_factory
from enum import Enum
import numpy as np 
import pandas as pd
from magicgui import magicgui


from napari.qt.threading import thread_worker

from superqt import QLabeledDoubleRangeSlider

from nyxus_napari.range_slider import RangeSlider

import dask
import psutil

import nyxus
from nyxus_napari import util

class Features(Enum):
    All = "*ALL*"
    Intensity = "*ALL_INTENSITY*"
    All_Morphology = "*ALL_MORPHOLOGY*"
    Basic_Morphology = "*BASIC_MORPHOLOGY*"
    GLCM = "*ALL_GLCM*"
    GLRM = "*ALL_GLRM*"
    GLSZM = "*ALL_GLSZM*"
    GLDM = "*ALL_GLDM*"
    NGTDM = "*ALL_NGTDM*"
    All_but_Gabor = "*ALL_BUT_GABOR*"
    All_but_GLCM= "*ALL_BUT_GLCM*"
    
class FeaturesWidget(QWidget):
    
    
    @QtCore.Slot(QtWidgets.QTableWidgetItem)
    def onClicked(self, it):
        print('clicked')
        state = not it.data(SelectedRole)
        it.setData(SelectedRole, state)
        it.setBackground(
            QtGui.QColor(100, 100, 100) if state else QtGui.QColor(0, 0, 0)
        )

class NyxusNapari:
    
    def __init__(
        self,
        viewer: napari.Viewer,
        intensity: Image, 
        segmentation: Image,
        features: Features,
        save_to_csv: bool = True,
        output_path: "str" = "",
        neighbor_distance: float = 5.0,
        pixels_per_micron: float = 1.0,
        coarse_gray_depth: int = 256, 
        use_CUDA_Enabled_GPU: bool = False,
        gpu_id: int = 0):
    
        self.viewer = viewer
        self.intensity = intensity
        self.segmentation = segmentation
        self.save_to_csv = save_to_csv
        self.output_path = output_path
    
        self.nyxus_object = None
        self.result = None

        self.current_label = 0
        self.seg = self.segmentation.data
        self.labels = np.zeros_like(self.seg)
        self.colormap = np.zeros_like(self.seg)
        self.colormap_added = False
        
        self.labels_added = False
        
        if (use_CUDA_Enabled_GPU):
            import subprocess
            
            try:
                subprocess.check_output('nvidia-smi')
                show_info('Nvidia GPU detected')
            except Exception: # this command not being found can raise quite a few different errors depending on the configuration
                show_info('No Nvidia GPU found. The machine must have a CUDA enable Nvidia GPU with drivers installed.')
                return
                
            self.nyxus_object = nyxus.Nyxus([features.value], 
                                neighbor_distance=neighbor_distance, 
                                pixels_per_micron=pixels_per_micron, 
                                coarse_gray_depth=coarse_gray_depth,
                                using_gpu = gpu_id)
            
        else:
            self.nyxus_object = nyxus.Nyxus([features.value], 
                                    neighbor_distance=neighbor_distance, 
                                    pixels_per_micron=pixels_per_micron, 
                                    coarse_gray_depth=coarse_gray_depth,
                                    using_gpu = -1)
        
        @segmentation.mouse_drag_callbacks.append
        def clicked_roi(layer, event):
            coords = np.round(event.position).astype(int)
            value = layer.data[coords[0]][coords[1]]
            if (value == 0):
                return
            self.table.selectRow(value)
            
        @intensity.mouse_drag_callbacks.append
        def clicked_roi(layer, event):
            coords = np.round(event.position).astype(int)
            value = segmentation.data[coords[0]][coords[1]]
            if (value == 0):
                return
            self.table.selectRow(value)
    
    
    def calculate(self):  
        self.result = self.nyxus_object.featurize(self.intensity.data, self.segmentation.data)
        if (self.save_to_csv):
            show_info("Saving results to " + self.output_path + "out.csv")
            self.result.to_csv(self.output_path + 'out.csv', sep='\t', encoding='utf-8')
   

    def add_features_table(self):
        # Create window for the DataFrame viewer
        self.win = FeaturesWidget()
        scroll = QScrollArea()
        layout = QVBoxLayout()
        self.table = QTableWidget()
        scroll.setWidget(self.table)
        layout.addWidget(self.table)
        self.win.setLayout(layout)    
        self.win.setWindowTitle("Feature Results")

        # Add DataFrame to widget window
        self.table.setColumnCount(len(self.result.columns))
        self.table.setRowCount(len(self.result.index))
        self.table.setHorizontalHeaderLabels(self.result.columns)
        for i in range(len(self.result.index)):
            for j in range(len(self.result.columns)):
                self.table.setItem(i,j,QTableWidgetItem(str(self.result.iloc[i, j])))
                
        self.table.cellClicked.connect(self.cell_was_clicked)
        self.table.horizontalHeader().sectionClicked.connect(self.onHeaderClicked)

        # add DataFrame to Viewer
        self.viewer.window.add_dock_widget(self.win)
    
    def highlight_value(self, value):

        removed = False

        for ix, iy in np.ndindex(self.seg.shape):

            if (int(self.seg[ix, iy]) == int(value)):

                if (self.labels[ix, iy] != 0):
                    self.labels[ix, iy] = 0
                else:
                    self.labels[ix, iy] = int(value)
        
        if (not removed):
            self.current_label += 1
            
        if (not self.labels_added):
            self.viewer.add_labels(np.array(self.labels).astype('int8'), name="Selected ROI")
            self.labels_added = True
        else:
            self.viewer.layers["Selected ROI"].data = np.array(self.labels).astype('int8')

            
    def cell_was_clicked(self, event):
        current_column = self.table.currentColumn()
        
        if(current_column == 2):
            current_row = self.table.currentRow()
            cell_value = self.table.item(current_row, current_column).text()
            
            self.highlight_value(cell_value)
    
    def onHeaderClicked(self, logicalIndex):
        self.create_feature_color_map(logicalIndex)
        
    def create_feature_color_map(self, logicalIndex):
        self.colormap = np.zeros_like(self.seg)
        
        labels = self.result.iloc[:,2]
        values = self.result.iloc[:,logicalIndex]
        label_values = pd.Series(values, index=labels).to_dict()
        
        for ix, iy in np.ndindex(self.seg.shape):
            
            if (self.seg[ix, iy] != 0):
                if(np.isnan(label_values[int(self.seg[ix, iy])])):
                    continue
                
                self.colormap[ix, iy] = label_values[int(self.seg[ix, iy])]
        
        if (not self.colormap_added):
            self.viewer.add_image(np.array(self.colormap), name="Colormap")
            self.colormap_added = True
            
        else:
            self.viewer.layers["Colormap"].data = np.array(self.colormap)
        
        if (self.slider_added):
            self._update_slider([min_value, max_value])
        else:
            self._add_range_slider(min_value, max_value, self.result.columns[logicalIndex])
        
    
    
    def _get_label_from_range(self, min_bound, max_bound):

        for ix, iy in np.ndindex(self.colormap.shape):
            
            if (self.seg[ix, iy] != 0):
                
                if(np.isnan(self.label_values[int(self.seg[ix, iy])])):
                    continue
                
                value = self.label_values[int(self.seg[ix, iy])]
                
                if (value <= max_bound and value >= min_bound):
                    self.labels[ix, iy] = self.seg[ix, iy]
                    self.colormap[ix, iy] = value
                else:
                    self.labels[ix, iy] = 0
                    self.colormap[ix, iy] = 0
                
        if (not self.colormap_added):
            self.viewer.add_image(np.array(self.colormap), name="Colormap")
            self.colormap_added = True
            
        else:
            self.viewer.layers["Colormap"].data = np.array(self.colormap)
            
             
        if (not self.labels_added):
            self.viewer.add_labels(np.array(self.labels).astype('int8'), name="Selected ROI")
            self.labels_added = True
            
        else:
            self.viewer.layers["Selected ROI"].data = np.array(self.labels).astype('int8')
            
    
    def _add_range_slider(self, min_value, max_value, name):
        min_value = util.round_down_to_5_sig_figs(min_value)
        max_value = util.round_up_to_5_sig_figs(max_value)
        
        if (self.slider_added):
            self.slider.setRange(min_value, max_value)
            self.dock_widget.setWindowTitle(name)

        else:
            self.slider = RangeSlider(QtCore.Qt.Horizontal)
            self.slider.setMinimumHeight(30)
            self.slider.setMinimum(min_value)
            self.slider.setMaximum(max_value)
            self.slider.setLow(min_value)
            self.slider.setHigh(max_value)
            #self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            self.slider.sliderMoved.connect(self._update_slider)
            
            layout = QVBoxLayout()
            self.name_label = QLabel(name)
            self.name_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.name_label)
            
            layout.addWidget(self.slider)
            
            """
            slider_hbox = QHBoxLayout()
            slider_hbox.setContentsMargins(0, 0, 0, 0)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            label_minimum = QLabel(alignment=QtCore.Qt.AlignLeft)
            #self.slider.minimumChanged.connect(label_minimum.setNum)
            label_maximum = QLabel(alignment=QtCore.Qt.AlignRight)
            #self.slider.maximumChanged.connect(label_maximum.setNum)
            layout.addWidget(self.slider)
            layout.addLayout(slider_hbox)
            slider_hbox.addWidget(label_minimum, QtCore.Qt.AlignLeft)
            slider_hbox.addWidget(label_maximum, QtCore.Qt.AlignRight)
            layout.addStretch()
            """
            
            widget = QWidget()
            widget.setLayout(layout)
            self.dock_widget = self.viewer.window.add_dock_widget(widget)
            self.dock_widget.setWindowTitle(name)
            self.dock_widget.resizeEvent = self._update_slider_size
            
            self.min_box = QLineEdit(str(min_value))
            self.min_box.setReadOnly(True)
            self.max_box = QLineEdit(str(max_value))
            self.max_box.setReadOnly(True)

            #self.value_box = QLineEdit(str(initial_value))
            #self.value_box.setReadOnly(True)
            
            hlayout = QHBoxLayout()
            hlayout.addWidget(self.min_box)
            hlayout.addWidget(self.max_box)
            layout.addLayout(hlayout)
            
            #self.dock_widget.addLayout(slider_vbox)
            
            # Add a label to the dock widget to display text at the top
            self.text = name
            self.label = QLabel("Adjust Range")
            self.label.setAlignment(Qt.AlignCenter)
            self.dock_widget.setTitleBarWidget(self.label)

            self.slider_added = True
        
    def _update_slider_size(self, event):
        print(event)
        #dock_widget = self.slider.parent()
        #slider_size = dock_widget.size()
        #self.slider.resize(event.size())
        
        
        
        
        
        # Calculate the size of the handles based on the slider size
        #handle_size = int(slider_size.height() * 0.8)
        #handle_margin = int((slider_size.height() - handle_size) / 2)
        
        #self.slider.setStyleSheet('''
        #    QLabeledDoubleRangeSlider::handle:horizontal {
        #        height: 1000px;
        #        width: 1000px;
        #    }
                                  
        #''')
        
        # Update the size of the handles by modifying their stylesheet
        """
        handle_stylesheet = (
            f"QLabeledDoubleRangeSlider::handle:horizontal {{"
            f"height: {handle_size}px;"
            f"margin-top: {handle_margin}px;"
            f"margin-bottom: {handle_margin}px;"
            "}"
        )
        self.slider.setStyleSheet(handle_stylesheet)
        """
        #self.slider.setFixedWidth(event.size().width())
        
        #self.slider.leftHandle().setFixedSize(handle_size, handle_size)
        #self.slider.rightHandle().setFixedSize(handle_size, handle_size)
        
    def _update_slider(self, event):
        print(event)
        #self._get_label_from_range(event[0], event[1])
        
    
    """
    def _add_range_slider(self, min_value, max_value):
        
        if (self.slider_added):
            self.slider.setRange(min_value, max_value)

        else:
            self.slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(min_value, max_value)
            self.slider.setValue([min_value, max_value])
            self.slider.valueChanged.connect(self._update_slider)
            
            self.viewer.window.add_dock_widget(self.slider)
            
            
            # Connect to the resizeEvent of the parent widget
            parent = self.viewer.window
            
            print(dir(self.viewer.window))
            
            #self.slider.resize(parent.size().width(), parent.size().height())

            #parent.resizeEvent = lambda event: self.slider.resize(parent.size().width(), parent.size().height())

            self.slider_added = True
        
    def _update_slider(self, event):

        self._get_label_from_range(event[0], event[1])
    """