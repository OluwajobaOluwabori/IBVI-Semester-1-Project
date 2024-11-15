import os, sys, time, pdb
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi  # pip install scipy
import scipy.signal as sig
import pandas as pd, math
import wfdb
from IPython.core.pylabtools import figsize
from IPython.display import display
from scipy.signal import periodogram,welch
from scipy.signal.windows import hamming, hann, boxcar
from scipy.signal import find_peaks

import openpyxl

from absl.logging import exception


from PyQt5.QtGui import QImage, QPixmap,QFont
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,QVBoxLayout,QHBoxLayout, QFrame
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
from numpy import *
from ProjectS1v1GUI import Ui_MainWindow
from PeriodogramWindow import Ui_Form as Ui_FormPeriodogram
from FFTWindow import Ui_Form as Ui_FormFFT
# from RotateWindow import Ui_Form as Ui_FormRotate
# from BrightnessWindow import Ui_Form as Ui_FormBright
# from SaturationWindow import Ui_Form as Ui_FormSaturation
b_Canvas = False

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.icon_name_widget.setHidden(True)


        self.ui.pbDataManagement1.clicked.connect(self.switch_to_dataManagementPage)
        self.ui.pbDataManagement2.clicked.connect(self.switch_to_dataManagementPage)
        self.ui.pbDataAnalysis1.clicked.connect(self.switch_to_DataAnalysisPage)
        self.ui.pbDataAnalysis2.clicked.connect(self.switch_to_DataAnalysisPage)
        self.ui.pbSpectrumAnalysis1.clicked.connect(self.switch_to_SpectrumAnalysisPage)
        self.ui.pbSpectrumAnalysis2.clicked.connect(self.switch_to_SpectrumAnalysisPage)
        self.ui.pbImageProcessing1.clicked.connect(self.switch_to_ImageProcessingPage)
        self.ui.pbImageProcessing2.clicked.connect(self.switch_to_ImageProcessingPage)

        # Link the clicked action from the button to a function to read excel file
        self.ui.LoadBtnsignal.clicked.connect(self.loadDatasignal)
        self.ui.WindowcomboBox.currentTextChanged.connect(self.plotSignal)
        self.ui.ScalingcomboBox.currentTextChanged.connect(self.plotSignal)
        self.ui.NormcomboBox.currentTextChanged.connect(self.plotSignal)
        self.ui.pbValidate.clicked.connect(self.updateMainWindow)

        # Link the clicked action from the button to a function to read excel file
        self.ui.LoadBtn_excel.clicked.connect(self.loadExcelData)
        # Link the clicked action from the button to a function to plot data
        self.ui.pbPlot.clicked.connect(self.plotExcelData)
        # Link the combobox to a function when we select an item from the list
        self.ui.cbX.currentIndexChanged.connect(self.cbXvaluechanged)
        self.ui.cbY.currentIndexChanged.connect(self.cbYvaluechanged)

        # Button can't be clicked because no reason to if Excel not yet loaded
        self.ui.pbPlot.setEnabled(False)

        # Change the default name of the tab1 and tab2
        self.ui.tabWidget.setTabText(0, "Plot")
        self.ui.tabWidget.setTabText(1, "Data")
        # Force the focus on the first tab
        self.ui.tabWidget.setCurrentIndex(0)

        # Link the clicked action from the button to a function to plot data
        # self.ui.pbPlot.clicked.connect(self.plotData)
        # self.ui.pbFFT.clicked.connect(self.showFFTWindow)
        # self.ui.pbPeriodogram.clicked.connect(self.showPeriodogramWindow)


        # Button can't be clicked because no reason to if Excel not yet loaded
        # self.ui.pbPlot.setEnabled(False)
        # self.ui.pbFFT.setEnabled(False)
        # self.ui.pbPeriodogram.setEnabled(False)

        self.canvas = None
        self.toolbar = None







    def switch_to_dataManagementPage(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    def switch_to_DataAnalysisPage(self):
        self.ui.stackedWidget.setCurrentIndex(1)
    def switch_to_SpectrumAnalysisPage(self):
        self.ui.stackedWidget.setCurrentIndex(2)
        self.ui.LoadBtnsignal.clicked.connect(self.loadDatasignal)
        # self.ui.pbPeriodogram.clicked.connect(self.Periodogram)
    def switch_to_ImageProcessingPage(self):
        self.ui.stackedWidget.setCurrentIndex(3)


    def loadDatasignal(self):
        try:
            myDlg = QFileDialog.getOpenFileName(None, 'OpenFile', "", "Record Files (*.dat)")

            self.myPath = myDlg[0][:-4] # Path + file + extension
            print(self.myPath)
            FileNameWithExtension = QFileInfo(myDlg[0]).fileName()  # Just the file + extension
            print(self.myPath)

            if myDlg[0] == myDlg[1] == '':
                # No file selected or cancel button clicked - so do nothing
                pass
            else:
                # Read and extract values from Excel
                self.data = wfdb.rdrecord(self.myPath)
                self.signal = self.data.p_signal[:, 0]
                self.data.fs = self.data.fs
                self.data.n_sig = self.data.n_sig

                # Plot the loaded data and initial FFT and periodogram
                self.plotSignal()
        except Exception as e:
            print(e)
    def plotSignal(self):
            # Clear existing canvas and toolbar if they exist
            try:
                layout = self.ui.mplwindow.layout()
                if self.canvas:
                    layout.removeWidget(self.canvas)
                    self.canvas.deleteLater()
                    self.canvas = None
                if self.toolbar:
                    layout.removeWidget(self.toolbar)
                    self.toolbar.deleteLater()
                    self.toolbar = None
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Create one figure with 3 subplots
                ax[0].plot(self.signal)
                ax[0].set_title("Loaded Signal Data")
                ax[0].set_xlabel("Samples")
                ax[0].set_ylabel("Amplitude")
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
                #
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks


                # self.canvas = FigureCanvas(fig)
                # layout = self.ui.mplwindow.layout()
                # layout.addWidget(self.canvas)
                # self.canvas.draw()
                # self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
                # layout.addWidget(self.toolbar)

                # Periodogram
                scaling_value = self.ui.ScalingcomboBox.currentText()
                window_value = self.ui.WindowcomboBox.currentText()

                f, Pxx = periodogram(self.signal, fs=self.data.fs, scaling=scaling_value,
                                     window=window_value)
                Pxx_dB = 10 * np.log10(Pxx)  # Convert power to dB

                # Plot within Periodogramlabel using a FigureCanvas
                # fig = Figure(figsize(6, 6))
                # ax = fig.add_subplot(132)
                ax[1].clear()
                ax[1].plot(f, Pxx_dB)
                ax[1].set_title(f'Periodogram of the Signal\n Window Type = {window_value}, Scaling = {scaling_value}'
                          )
                ax[1].set_xlabel('Frequency (Hz)')
                if scaling_value == 'spectrum':
                    ax[1].set_ylabel('Power(dB/Hz)')
                else:
                    ax[1].set_ylabel('Power/Frequency (dB/Hz)')
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
                #
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks
                # Embed the plot in the QLabel (Periodogramlabel)
                # self.canvas = FigureCanvas(fig)
                # layout = self.ui.mplwindow.layout()
                # layout.addWidget(self.canvas)
                # self.canvas.draw()
                # self.ui.verticalLayout_5.addWidget(self.toolbar)


                # FFT
                # Get the selected window and scaling values
                norm_value = self.ui.NormcomboBox.currentText()
                # Step 1: Calculate the FFT
                fft_values = np.fft.fft(self.signal, norm=norm_value)
                # Step 2: Calculate the frequencies corresponding to the FFT values
                n = len(self.signal)
                frequencies = np.fft.fftfreq(n, d=1 / self.data.fs)

                # Step 3: Get the positive frequencies and the corresponding FFT values
                positive_frequencies = frequencies[:n // 2]
                positive_fft_values = np.abs(fft_values[:n // 2])  # Magnitude of the FFT

                # Step 4: Plot the FFT
                # fig = Figure(figsize(6, 6))
                # ax = fig.add_subplot(133)
                ax[2].clear()
                ax[2].plot(positive_frequencies, positive_fft_values)
                ax[2].set_title(f'Frequency Spectrum of the Signal\n Norm = {norm_value}')
                ax[2].set_xlabel('Frequency (Hz)')
                ax[2].set_ylabel('Amplitude')
                fig.tight_layout()
                self.canvas = FigureCanvas(fig)
                layout = self.ui.mplwindow.layout()
                layout.addWidget(self.canvas)
                self.canvas.draw()
                self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
                # layout.addWidget(self.toolbar)
                # Add toolbar at the top
                # self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow)
                layout.insertWidget(0, self.toolbar)  # Insert at the top of the layout

                # self.canvas = FigureCanvas(fig)
                # layout = self.ui.mplwindow.layout()
                # layout.addWidget(self.canvas)
                # self.canvas.draw()
                # self.ui.verticalLayout_5.addWidget(self.toolbar)

            #     # Button to plot is now clickable
            # self.ui.pbPlot.setEnabled(True)
            # self.ui.pbFFT.setEnabled(True)
            # self.ui.pbPeriodogram.setEnabled(True)
            # self.ui.pbPlot.setEnabled(True)
            except Exception as e:
             print(e)

    def loadExcelData(self):
        try:
            myDlg = QFileDialog.getOpenFileName(None, 'OpenFile', "", "Excel Files (*.xls *.xlsx *.csv)")
            self.myPath = myDlg[0]  # Path + file + extension
            FileNameWithExtension = QFileInfo(myDlg[0]).fileName()  # Just the file + extension

            if myDlg[0] == myDlg[1] == '':
                # No file selected or cancel button clicked - so do nothing
                pass
            else:
                # Read and extract values from Excel
                # try:
                #     self.data = pd.read_csv(self.myPath)
                # except:
                self.data = pd.read_excel(self.myPath)

                # Change the text of the label to show the name of the file
                self.ui.lblExcel.setText(FileNameWithExtension)
                # Change the text of the second tab to show the path of the file
                self.ui.tabWidget.setTabText(1, self.myPath)
                # Extract headers of all columns and convert to a list
                self.df = pd.DataFrame(self.data)
                self.headers = self.df.columns.tolist()
                print(self.headers)
                # Add the list to all comboboxes
                # self.ui.cbX.addItems(self.headers)
                # self.ui.cbY.addItems(self.headers)
                # self.ui.cbMap.addItem("")
                # self.ui.cbMap.addItems(self.headers)
                # Prepare the table that will contain the Excel
                # Create N rows, N columns and add the headers
                self.ui.tableWidget.setRowCount(self.df.shape[0])
                self.ui.tableWidget.setColumnCount(self.df.shape[1])
                self.ui.tableWidget.setHorizontalHeaderLabels(self.headers)
                # A double loop to read the values and add them to the table
                for row in self.df.iterrows():
                    values = row[1]
                    for col, value in enumerate(values):
                        tableItem = QTableWidgetItem(str(value))
                        self.ui.tableWidget.setItem(row[0], col, tableItem)
                # Button to plot is now clickable
                self.ui.pbPlot.setEnabled(True)

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print(myDlg)
        except ValueError:
            print("Value Error.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    def cbXvaluechanged(self):
        X = self.ui.cbX.currentText()

        self.xmin = self.ui.sbXmin.setValue(self.data[X].min())
        self.xmax = self.ui.sbXmax.setValue(self.data[X].max())

    def cbYvaluechanged(self):
        Y = self.ui.cbY.currentText()

        self.ymin = self.ui.sbYmin.setValue(self.data[Y].min())
        self.ymax = self.ui.sbYmax.setValue(self.data[Y].max())


    def plotExcelData(self):
        global b_Canvas

        # Remove the previous plot
        if b_Canvas == True:
            self.rm_mpl()

        # Extract data and headers
        data = self.data
        X = self.ui.cbX.currentText()
        Y = self.ui.cbY.currentText()
        Cmap = self.ui.cbMap.currentText()

        # Extract the values from the 4 boxes to use them for the double slicing
        xmin = self.ui.sbXmin.value()
        xmax = self.ui.sbXmax.value()
        ymin = self.ui.sbYmin.value()
        ymax = self.ui.sbYmax.value()

        # Extract only the data respecting the 4 conditions
        NewData = data.loc[(data[X] > xmin) & (data[X] < xmax) & (data[Y] > ymin) & (data[Y] < ymax)]
        # It's one way, we could also use numpy, for example:
        # np.where((np.array(data[X]) > xmin) & (np.array(data[X]) < xmax))

        # Save new data in old data variable so we don't have to modify the plot part
        data = NewData

        # Prepare the plot
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.ui.verticalLayout.addWidget(self.canvas)
        ax1f1 = fig.add_subplot(111)
        # Reformat the coordinates near the toolbar to not show values like 2.425e+04 but 24250 for example
        ax1f1.format_coord = lambda x, y: "x=%1.4f, y=%1.4f"%(x, y)

        # If no cmap then will do a normal plot (X,Y)
        if self.ui.cbMap.currentText() == "":
            ax1f1.plot(data[X], data[Y], 'o', markersize=2)
        elif self.ui.cbMap.currentText() == "Press":
            # If selecting pressure, swap the colormap so high pressure is in red instead of blue
            ax1f1.bar(data[X], data[Y], cmap='jet_r', c=data[Cmap])
        else:
            ax1f1.bar(data[X], data[Y], cmap='jet', c=data[Cmap])

        # Add a title for the figure and axis
        ax1f1.set_title('Title')
        ax1f1.set_xlabel("X", fontsize=14)
        ax1f1.set_ylabel("Y", fontsize=14)

        # If checkbox is checked, then Y axis is inverted
        if self.ui.checkBox.isChecked():
            ax1f1.invert_yaxis()

        # Draw everything and add toolbar
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
        self.ui.verticalLayout.addWidget(self.toolbar)
        b_Canvas = True

    def rm_mpl(self):
        global b_Canvas
        # Remove plot part and toolbar
        self.ui.verticalLayout.removeWidget(self.canvas)
        self.canvas.close()
        self.ui.verticalLayout.removeWidget(self.toolbar)
        self.toolbar.close()
        b_Canvas = False



    def updateMainWindow(self):
        return

if __name__ == '__main__':
    import sys

    QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    # To avoid weird behaviors (smaller items, ...) on big resolution screens
    app.setStyle("fusion")
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())


