# ----- Imports -------------------------------------------------------

# Standard Imports
import sys
import numpy as np
import time
import math
import struct
import os
import string
import serial
import serial.tools.list_ports
import statistics
import warnings
import random

# PyQt5 Imports
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog, QButtonGroup, QFormLayout)
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.pgcollections import OrderedDict
from gl_classes import GLTextItem

# Local File Imports
from gui_threads import *
from gui_parser import uartParser
from graphUtilities import *
from gui_common import *

# ----- Defines -------------------------------------------------------
compileGui = 0

# Only when compiling
if (compileGui):
    from fbs_runtime.application_context.PyQt5 import ApplicationContext


# Create a list of N distict colors, visible on the black GUI background, for our tracks
# The format for a single color is (r,g,b,a) -> normalized from 0-255 to 0-1
# LUT based on Kelly's 22 Colors of Max Contrast, slightly adjusted for better visibility on black background (https://sashamaps.net/docs/resources/20-colors/)
# Only the first 21 colors are guaranteed to be highly distinct. After that colors are generated, but not promised to be visually distinct.
def get_trackColors(n):
    # Modified LUT of Kelly's 22 Colors of Max Contrast
    modKellyColors = [
        # (255, 255, 255, 255),   # White
        # (  0,   0,   0, 255),   # Black
        # (169, 169, 169, 255),   # Gray
        (230,  25,  75, 255),   # Red
        ( 60, 180,  75, 255),   # Green
        (255, 225,  25, 255),   # Yellow
        ( 67,  99, 216, 255),   # Blue
        (245, 130,  49, 255),   # Orange
        (145,  30, 180, 255),   # Purple
        ( 66, 212, 244, 255),   # Cyan
        (240,  50, 230, 255),   # Magenta
        (191, 239,  69, 255),   # Lime
        (250, 190, 212, 255),   # Pink
        ( 70, 153, 144, 255),   # Teal
        (220, 190, 255, 255),   # Lavender
        (154,  99,  36, 255),   # Brown
        (255, 250, 200, 255),   # Beige
        (128,   0,   0, 255),   # Maroon
        (170, 255, 195, 255),   # Mint
        (128, 128,   0, 255),   # Olive
        (255, 216, 177, 255),   # Apricot
        (  0,   0, 117, 255)    # Navy
    ]
    
    # Generate normalized version of Kelly colors
    modKellyColorsNorm = []
    for tup in modKellyColors: 
        modKellyColorsNorm.append(tuple(ti/255 for ti in tup))
    
    # Create the output color list
    trackColorList = []
    for i in range(n):
        # If within the length of the LUT, just grab values
        if i < len(modKellyColorsNorm):
            trackColorList.append(modKellyColorsNorm[i])
        # Otherwise, generate a color from the average of two randomly selected colors, and add the new color to the list
        else:  
            (r_2, g_2, b_2, _) = modKellyColorsNorm[random.randint(0,len(modKellyColorsNorm)-1)]
            (r_1, g_1, b_1, _) = modKellyColorsNorm[random.randint(0,len(modKellyColorsNorm)-1)]
            r_gen = (r_2 + r_1) / 2
            g_gen = (g_2 + g_1) / 2
            b_gen = (b_2 + b_1) / 2
            modKellyColorsNorm.append((r_gen, g_gen, b_gen , 1.0))
            trackColorList.append(    (r_gen, g_gen, b_gen , 1.0))

    return trackColorList


class Window(QDialog):
    def __init__(self, parent=None, size=[]):
        super(Window, self).__init__(parent)
        # set window toolbar options, and title
        self.setWindowFlags(
            Qt.Window |
            Qt.CustomizeWindowHint |
            Qt.WindowTitleHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("mmWave Industrial Visualizer")

        if (0): #set to 1 to save terminal output to logFile, set 0 to show terminal output
            ts = time.localtime()
            terminalFileName = str('logData/logfile_'+ str(ts[2]) + str(ts[1]) + str(ts[0]) + '_' + str(ts[3]) + str(ts[4]) +'.txt')
            sys.stdout = open(terminalFileName, 'w')

        print('Python is ', struct.calcsize("P")*8, ' bit')
        print('Python version: ', sys.version_info)

        # TODO bypass serial read function to also log to a file

        self.frameTime = 50
        self.graphFin = 1
        self.hGraphFin = 1
        self.threeD = 1
        self.lastFramePoints = np.zeros((5,1))
        self.plotTargets = 1
        self.frameNum = 0
        self.profile = {'startFreq': 60.25, 'numLoops': 64, 'numTx': 3, 'sensorHeight':3, 'maxRange':10, 'az_tilt':0, 'elev_tilt':0}
        self.sensorHeight = 1.5
        self.numFrameAvg = 10
        self.configSent = 0
        self.previousFirstZ = -1
        self.yzFlip = 0
        #timer to reset fall detected message
        self.fallTimer = QTimer()
        self.fallTimer.setSingleShot(True)
        self.fallTimer.timeout.connect(self.resetFallText)
        self.fallResetTimerOn = 0
        self.fallThresh = -0.22

        self.trackColorMap = None
        self.prevConfig = DEMO_NAME_OOB
        self.vitalsPatientData = []

        #color gradients
        # TODO Simplify color gradients
        self.Gradients = OrderedDict([
    ('bw', {'ticks': [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))], 'mode': 'rgb'}),
    ('hot', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
    ('jet', {'ticks': [(1, (166, 0, 0, 255)), (0.32247191011235954, (0, 255, 255, 255)), (0.11348314606741573, (0, 68, 255, 255)), (0.6797752808988764, (255, 255, 0, 255)), (0.902247191011236, (255, 0, 0, 255)), (0.0, (0, 0, 166, 255)), (0.5022471910112359, (0, 255, 0, 255))], 'mode': 'rgb'}),
    ('summer', {'ticks': [(1, (255, 255, 0, 255)), (0.0, (0, 170, 127, 255))], 'mode': 'rgb'} ),
    ('space', {'ticks': [(0.562, (75, 215, 227, 255)), (0.087, (255, 170, 0, 254)), (0.332, (0, 255, 0, 255)), (0.77, (85, 0, 255, 255)), (0.0, (255, 0, 0, 255)), (1.0, (255, 0, 127, 255))], 'mode': 'rgb'}),
    ('winter', {'ticks': [(1, (0, 255, 127, 255)), (0.0, (0, 0, 255, 255))], 'mode': 'rgb'}),
    ('spectrum2', {'ticks': [(1.0, (255, 0, 0, 255)), (0.0, (255, 0, 255, 255))], 'mode': 'hsv'}),
    ('heatmap', {'ticks': [ (1, (255, 0, 0, 255)), (0, (131, 238, 255, 255))], 'mode': 'hsv'})
])
        cmap = 'heatmap'
        if (cmap in self.Gradients):
            self.gradientMode = self.Gradients[cmap]
        self.zRange = [-3, 3]
        self.plotHeights = 1
        # Gui size
        if (size):
            left = 50
            top = 50
            width = math.ceil(size.width()*0.9)
            height = math.ceil(size.height()*0.9)
            self.setGeometry(left, top, width, height)
        
        # Persistent point cloud
        self.previousClouds = []

        self.hearPlotData = []
        self.breathPlotData = []

        # Set up graph pyqtgraph
        self.init3dGraph()
        self.initColorGradient()

        # Add connect options
        self.initConnectionPane()
        self.initStatsPane()
        self.initPlotControlPane()
        self.initConfigPane()
        self.initSensorPositionPane()
        self.initBoundaryBoxPane()
        self.initVitalsPlots()

        # Set the layout
        # Create tab for different graphing options
        self.graphTabs = QTabWidget()
        self.graphTabs.addTab(self.pcplot, '3D Plot')
        self.graphTabs.currentChanged.connect(self.whoVisible)

        gridlay = QGridLayout()
        gridlay.addWidget(self.comBox, 0,0,1,1)
        gridlay.addWidget(self.statBox, 1,0,1,1)
        gridlay.addWidget(self.configBox,2,0,1,1)
        gridlay.addWidget(self.plotControlBox,3,0,1,1)
        gridlay.addWidget(self.spBox,4,0,1,1)
        gridlay.addWidget(self.boxTab,5,0,1,1)
        gridlay.addWidget(self.graphTabs,0,1,6,1)
        gridlay.addWidget(self.colorGradient, 0, 2, 6, 1)
        gridlay.addWidget(self.vitalsPane, 0, 3, 6, 1)
        self.vitalsPane.setVisible(False)
        gridlay.setColumnStretch(0,1)
        gridlay.setColumnStretch(1,3)
        self.setLayout(gridlay)


    def initConnectionPane(self):
        self.comBox = QGroupBox('Connect to Com Ports')
        self.cliCom = QLineEdit('')
        self.dataCom = QLineEdit('')
        self.connectStatus = QLabel('Not Connected')
        self.connectButton = QPushButton('Connect')
        self.connectButton.clicked.connect(self.connectCom)
        self.configType = QComboBox()

        # TODO Add fall detection support
        # TODO Add replay support
        self.configType.addItems([DEMO_NAME_OOB, DEMO_NAME_3DPC, DEMO_NAME_VITALS, DEMO_NAME_MT, DEMO_NAME_LRPD, DEMO_NAME_SOD])
        self.configType.currentIndexChanged.connect(self.onChangeConfigType)
        self.comLayout = QGridLayout()
        self.comLayout.addWidget(QLabel('CLI COM:'),0,0)
        self.comLayout.addWidget(self.cliCom,0,1)
        self.comLayout.addWidget(QLabel('DATA COM:'),1,0)
        self.comLayout.addWidget(self.dataCom,1,1)
        self.comLayout.addWidget(QLabel('Demo:'),2,0)
        self.comLayout.addWidget(self.configType,2,1)
        self.comLayout.addWidget(self.connectButton,3,0)
        self.comLayout.addWidget(self.connectStatus,3,1)
        self.comBox.setLayout(self.comLayout)

        # Find all Com Ports
        serialPorts = list(serial.tools.list_ports.comports())

        # Find default CLI Port and Data Port
        for port in serialPorts:
            if (CLI_XDS_SERIAL_PORT_NAME in port.description or CLI_SIL_SERIAL_PORT_NAME in port.description):
                print(f'\tCLI COM Port found: {port.device}')
                comText = port.device
                comText = comText.replace("COM", "")
                self.cliCom.setText(comText)

            elif (DATA_XDS_SERIAL_PORT_NAME in port.description or DATA_SIL_SERIAL_PORT_NAME in port.description):
                print(f'\tData COM Port found: {port.device}')
                comText = port.device
                comText = comText.replace("COM", "")
                self.dataCom.setText(comText)


    def initStatsPane(self):
        self.statBox = QGroupBox('Statistics')
        self.frameNumDisplay = QLabel('Frame: 0')
        self.plotTimeDisplay = QLabel('Average Plot Time: 0 ms')
        self.numPointsDisplay = QLabel('Points: 0')
        self.numTargetsDisplay = QLabel('Targets: 0')
        self.statsLayout = QVBoxLayout()
        self.statsLayout.addWidget(self.frameNumDisplay)
        self.statsLayout.addWidget(self.plotTimeDisplay)
        self.statsLayout.addWidget(self.numPointsDisplay)
        self.statsLayout.addWidget(self.numTargetsDisplay)
        self.statBox.setLayout(self.statsLayout)


    def initPlotControlPane(self):
        self.plotControlBox = QGroupBox('Plot Controls')
        self.pointColorMode = QComboBox()
        self.pointColorMode.addItems([COLOR_MODE_SNR, COLOR_MODE_HEIGHT, COLOR_MODE_DOPPLER, COLOR_MODE_TRACK])
        self.plotTracks = QCheckBox('Plot Tracks')
        self.persistentFramesInput = QComboBox()
        self.persistentFramesInput.addItems(['1','2','3','4','5','6','7','8','9','10'])
        self.persistentFramesInput.setCurrentIndex(2)
        self.plotControlLayout = QFormLayout()
        self.plotControlLayout.addRow("Color Points By:",self.pointColorMode)
        self.plotControlLayout.addRow(self.plotTracks)
        self.plotControlLayout.addRow("# of Persistent Frames",self.persistentFramesInput)
        self.plotControlBox.setLayout(self.plotControlLayout)
        # Initialize button values
        self.plotTracks.setChecked(True)


    def initConfigPane(self):
        self.configBox = QGroupBox('Configuration')
        self.selectConfig = QPushButton('Select Configuration')
        self.sendConfig = QPushButton('Start and Send Configuration')
        self.start = QPushButton("Start without Send Configuration ")
        self.selectConfig.clicked.connect(self.selectCfg)
        self.sendConfig.clicked.connect(self.sendCfg)
        self.start.clicked.connect(self.startApp)     
        self.configTable = QTableWidget(5,2)
        # Set parameter names
        self.configTable.setItem(0,0,QTableWidgetItem('Radar Parameter'))
        self.configTable.setItem(0,1,QTableWidgetItem('Value'))
        self.configTable.setItem(1,0,QTableWidgetItem('Max Range'))
        self.configTable.setItem(2,0,QTableWidgetItem('Range Resolution'))
        self.configTable.setItem(3,0,QTableWidgetItem('Max Velocity'))
        self.configTable.setItem(4,0,QTableWidgetItem('Velocity Resolution'))
        self.configLayout = QVBoxLayout()
        self.configLayout.addWidget(self.selectConfig)
        self.configLayout.addWidget(self.sendConfig)
        self.configLayout.addWidget(self.start)
        self.configLayout.addWidget(self.configTable)       
        #self.configLayout.addStretch(1)
        self.configBox.setLayout(self.configLayout)


    def setControlLayout(self):
        self.controlBox = QGroupBox('Control')
        self.rangecfar = QSlider(Qt.Horizontal)
        self.azcfar = QSlider(Qt.Horizontal)
        self.snrthresh = QSlider(Qt.Horizontal)
        self.pointsthresh = QSlider(Qt.Horizontal)
        self.gatinggain = QSlider(Qt.Horizontal)
        self.controlLayout = QVBoxLayout()
        self.rangelabel = QLabel('Range CFAR Threshold: ')
        self.azlabel = QLabel('Azimuth CFAR Threshold: ')
        self.snrlabel = QLabel('SNR Threshold: ')
        self.pointslabel = QLabel('Points Threshold: ')
        self.gatinglabel = QLabel('Gating Gain: ')
        self.controlLayout.addWidget(self.rangelabel)
        self.controlLayout.addWidget(self.rangecfar)
        self.controlLayout.addWidget(self.azlabel)
        self.controlLayout.addWidget(self.azcfar)
        self.controlLayout.addWidget(self.snrlabel)
        self.controlLayout.addWidget(self.snrthresh)
        self.controlLayout.addWidget(self.pointslabel)
        self.controlLayout.addWidget(self.pointsthresh)
        self.controlLayout.addWidget(self.gatinglabel)
        self.controlLayout.addWidget(self.gatinggain)
        self.controlBox.setLayout(self.controlLayout)


    # Boundary box control section
    def setBoxControlLayout(self, name):
        # Set up one boundary box control
        boxControl = QGroupBox(name)
        
        description = QLabel('')
        # Input boxes
        lx = QLineEdit('-6')
        rx = QLineEdit('6')
        ny = QLineEdit('0')
        fy = QLineEdit('6')
        bz = QLineEdit('-6')
        tz = QLineEdit('6')
        enable = QCheckBox()

        # Set up color options
        color = QComboBox()
        color.addItem('Blue', 'b')
        color.addItem('Red', 'r')
        color.addItem('Green', 'g')
        color.addItem('Yellow', 'y')
        color.addItem('Cyan', 'c')
        color.addItem('Magenta', 'm')
        # color.addItem('Black', 'k')
        color.addItem('White', 'w')
        
        boxConLayout = QGridLayout()

        boxConLayout.addWidget(QLabel('Description:'),0,0,1,1)
        boxConLayout.addWidget(description,0,1,1,2)
        boxConLayout.addWidget(QLabel('Left X'),1,0,1,1)
        boxConLayout.addWidget(lx,1,1,1,1)
        boxConLayout.addWidget(QLabel('Right X'),1,2,1,1)
        boxConLayout.addWidget(rx,1,3,1,1)
        boxConLayout.addWidget(QLabel('Near Y'),2,0,1,1)
        boxConLayout.addWidget(ny,2,1,1,1)
        boxConLayout.addWidget(QLabel('Far Y'),2,2,1,1)
        boxConLayout.addWidget(fy,2,3,1,1)
        boxConLayout.addWidget(QLabel('Bottom Z'),3,0,1,1)
        boxConLayout.addWidget(bz,3,1,1,1)
        boxConLayout.addWidget(QLabel('Top Z'),3,2,1,1)
        boxConLayout.addWidget(tz,3,3,1,1)
        boxConLayout.addWidget(QLabel('Color'),4,0,1,1)
        boxConLayout.addWidget(color,4,1,1,1)
        boxConLayout.addWidget(QLabel('Enable Box'),4,2,1,1)
        boxConLayout.addWidget(enable,4,3,1,1)
        boxControl.setLayout(boxConLayout)
        boundList = [lx,rx,ny,fy,bz,tz]

        # Connect onchange listeners
        for text in boundList:
            text.textEdited.connect(self.onChangeBoundaryBox)
        enable.stateChanged.connect(self.onChangeBoundaryBox)
        color.currentIndexChanged.connect(self.onChangeBoundaryBox)
        # Return dictionary of all related controls for this box
        return {'name':name, 'boxCon':boxControl, 'boundList':boundList, 'checkEnable':enable, 'description':description, 'color':color}


    def initSensorPositionPane(self):
        self.az_tilt = QLineEdit('0')
        self.elev_tilt = QLineEdit('0')
        self.s_height = QLineEdit(str(self.profile['sensorHeight']))
        self.spLayout = QGridLayout()
        
        self.spLayout.addWidget(QLabel('Azimuth Tilt'),0,0,1,1)
        self.spLayout.addWidget(self.az_tilt,0,1,1,1)
        self.spLayout.addWidget(QLabel('Elevation Tilt'),1,0,1,1)
        self.spLayout.addWidget(self.elev_tilt,1,1,1,1)
        self.spLayout.addWidget(QLabel('Sensor Height'),2,0,1,1)
        self.spLayout.addWidget(self.s_height,2,1,1,1)
        
        self.spBox = QGroupBox('Sensor Position')
        self.spBox.setLayout(self.spLayout)
        self.s_height.textEdited.connect(self.onChangeSensorPosition)
        self.az_tilt.textEdited.connect(self.onChangeSensorPosition)
        self.elev_tilt.textEdited.connect(self.onChangeSensorPosition)
        # Force an update so that sensor is at default postion
        self.onChangeSensorPosition()


    def onChangeConfigType(self):
        newConfig = self.configType.currentText()
        print('Demo Changed to: ' + newConfig)
        
        # First, undo any changes that the last demo made
        # These should be the inverse of the changes made in 2nd part of this function

        # Undo OOB
        if (self.prevConfig == DEMO_NAME_OOB):
            # Unlock plot tracks
            self.plotTracks.setChecked(True)
            self.plotTracks.setDisabled(True)
        # Undo 3D People Counting
        elif (self.prevConfig == DEMO_NAME_3DPC):
            self.pointColorMode.setCurrentText(COLOR_MODE_SNR)
        # Undo Vitals
        elif (self.prevConfig == DEMO_NAME_VITALS):
            self.vitalsPane.setVisible(False)
            self.pointColorMode.setCurrentText(COLOR_MODE_SNR)
        # Undo Long Range People Detection
        elif (self.prevConfig == DEMO_NAME_LRPD):
            self.pointColorMode.setCurrentText(COLOR_MODE_SNR)
        # Undo Mobile Tracker
        elif (self.prevConfig == DEMO_NAME_MT):
            self.pointColorMode.setCurrentText(COLOR_MODE_SNR)
        # Undo Small Obstacle
        elif (self.prevConfig == DEMO_NAME_SOD):
            # Unlock boundary box config
            for box in self.boundaryBoxes:
                if ('occZone' in box['name']):
                    # Unlock each text field
                    for textBox in box['boundList']:
                        textBox.setDisabled(False)
                    # Unlock enable box
                    box['checkEnable'].setDisabled(False)
                    box['color'].setDisabled(False)

            # Unlock sensor position config
            self.spBox.setDisabled(False)
            
        
        # Now, apply any specific GUI changes for the new demo
        # Configure for Out of Box
        if (newConfig == DEMO_NAME_OOB):
            # Lock plot tracks off
            self.plotTracks.setChecked(False)
            self.plotTracks.setDisabled(False)
        # Configure for 3D People Counting
        elif (newConfig == DEMO_NAME_3DPC):
            self.pointColorMode.setCurrentText(COLOR_MODE_TRACK)
        # Configure For Vitals
        elif (newConfig == DEMO_NAME_VITALS):
            self.pointColorMode.setCurrentText(COLOR_MODE_TRACK)
            self.vitalsPane.setVisible(True)
        # Configure for Long Range People Detection
        elif (newConfig == DEMO_NAME_LRPD):
            self.pointColorMode.setCurrentText(COLOR_MODE_TRACK)
        # Configure for Mobile Tracker
        elif (newConfig == DEMO_NAME_MT):
            self.pointColorMode.setCurrentText(COLOR_MODE_TRACK)
        # Configure for Small Obstacle
        elif (newConfig == DEMO_NAME_SOD):
            # Lock boundary boxes for occ state machine
            for box in self.boundaryBoxes:
                if ('occZone' in box['name']):
                    # Lock each text field
                    for textBox in box['boundList']:
                        textBox.setDisabled(True)
                    # Lock enable box
                    box['checkEnable'].setDisabled(True)
                    box['color'].setDisabled(True)

            # Lock sensor position config
            self.s_height.setText('1')
            self.az_tilt.setText('0')
            self.elev_tilt.setText('0')
            self.spBox.setDisabled(True)
            self.onChangeSensorPosition()

        # Save this so that the next time we change configs we know what to undo
        self.prevConfig = newConfig


    # Gets called whenever the sensor position box is modified
    def onChangeSensorPosition(self):
        try:
            newHeight = float(self.s_height.text())
            newAzTilt = float(self.az_tilt.text())
            newElevTilt = float(self.elev_tilt.text())
        except:
            print("Error in gui_main.py: Failed to update sensor position")
            return
        command = "sensorPosition " + self.s_height.text() + " " + self.az_tilt.text() + " " + self.elev_tilt.text() + " \n"
        # self.cThread = sendCommandThread(self.parser,command)
        # self.cThread.start(priority=QThread.HighestPriority-2)

        # Update Profile info
        self.profile['sensorHeight'] = newHeight

        # Move evmBox to new position
        self.evmBox.resetTransform()
        self.evmBox.rotate(-1*newElevTilt,1,0,0)
        self.evmBox.rotate(-1*newAzTilt,0,0,1)
        self.evmBox.translate(0,0,newHeight)


    def initBoundaryBoxPane(self):
        # Set up all boundary box controls
        self.boundaryBoxes = []
        self.boxTab = QTabWidget()
        self.addBoundBox('pointBounds')
    

    # For live tuning when available
    def onChangeBoundaryBox(self):
        index = 0
        for box in self.boundaryBoxes:
            # Update dimensions of box
            try:
                xl = float(box['boundList'][0].text())
                xr = float(box['boundList'][1].text())
                yl = float(box['boundList'][2].text())
                yr = float(box['boundList'][3].text())
                zl = float(box['boundList'][4].text())
                zr = float(box['boundList'][5].text())
            except:
                # You get here if you enter an invalid number
                # When you enter a minus sign for a negative value, you will end up here before you type the full number
                pass
            boxLines = getBoxLines(xl,yl,zl,xr,yr,zr)
            boxColor = pg.glColor(box['color'].itemData(box['color'].currentIndex()))
            self.boundaryBoxViz[index].setData(pos=boxLines,color=boxColor,width=2,antialias=True,mode='lines')
            # Update visibility
            if (box['checkEnable'].isChecked()):
                self.boundaryBoxViz[index].setVisible(True)
                
            else:
                self.boundaryBoxViz[index].setVisible(False)
            index = index + 1


    def initVitalsPlots(self):
        self.vitalsPane = QGroupBox('Vital Signs')
        vitalsPaneLayout = QGridLayout()
        self.vitals = []

        for i in range(MAX_VITALS_PATIENTS):
            patientDict = {}
            patientName = 'Patient' + str(i+1)
            
            # Initialize the pane and layout
            patientPane = QGroupBox(patientName)
            patientPaneLayout = QGridLayout()

            # Set up basic labels so we can edit their appearance
            statusLabel = QLabel('Patient Status:')
            breathLabel = QLabel('Breath Rate:')
            heartLabel = QLabel('Heart Rate:')
            rangeBinLabel = QLabel('Range Bin:')

            # Set up patient vitals plot
            patientDict['plot'] = pg.PlotWidget()
            patientDict['plot'].setBackground('w')
            patientDict['plot'].showGrid(x=True,y=True)
            patientDict['plot'].invertX(True)
            patientDict['plot'].setXRange(0,NUM_VITALS_FRAMES_IN_PLOT,padding=0.01)
            patientDict['plot'].setYRange(-1,1,padding=0.1)
            patientDict['plot'].setMouseEnabled(False,False)
            patientDict['heartGraph'] = pg.PlotCurveItem(pen=pg.mkPen(width=3, color='r'))
            patientDict['breathGraph'] = pg.PlotCurveItem(pen=pg.mkPen(width=3, color='b'))
            patientDict['plot'].addItem(patientDict['heartGraph'])
            patientDict['plot'].addItem(patientDict['breathGraph'])

            # Set up all other patient data fields
            patientDict['breathRate'] = QLabel('Undefined')
            patientDict['heartRate'] = QLabel('Undefined')
            patientDict['status'] = QLabel('Undefined')
            patientDict['rangeBin'] = QLabel('Undefined')
            patientDict['name'] = patientName
            
            # Format text to make it attractive
            labelFont = QFont('Arial', 16)
            labelFont.setBold(True)
            dataFont = (QFont('Arial', 12))
            heartLabel.setFont(labelFont)
            breathLabel.setFont(labelFont)
            statusLabel.setFont(labelFont)
            rangeBinLabel.setFont(labelFont)
            patientDict['breathRate'].setStyleSheet('color: blue')
            patientDict['heartRate'].setStyleSheet('color: red')
            patientDict['status'].setFont(dataFont)
            patientDict['breathRate'].setFont(dataFont)
            patientDict['heartRate'].setFont(dataFont)
            patientDict['rangeBin'].setFont(dataFont)

            # Put the widgets into the layout
            patientPaneLayout.addWidget(patientDict['plot'],2,0,1,4)
            patientPaneLayout.addWidget(statusLabel,0,0,alignment=Qt.AlignHCenter)
            patientPaneLayout.addWidget(patientDict['status'],1,0,alignment=Qt.AlignHCenter)
            patientPaneLayout.addWidget(breathLabel,0,1,alignment=Qt.AlignHCenter)
            patientPaneLayout.addWidget(patientDict['breathRate'],1,1,alignment=Qt.AlignHCenter)
            patientPaneLayout.addWidget(heartLabel,0,2,alignment=Qt.AlignHCenter)
            patientPaneLayout.addWidget(patientDict['heartRate'],1,2,alignment=Qt.AlignHCenter)
            patientPaneLayout.addWidget(rangeBinLabel,0,3,alignment=Qt.AlignHCenter)
            patientPaneLayout.addWidget(patientDict['rangeBin'],1,3,alignment=Qt.AlignHCenter)

            patientPane.setLayout(patientPaneLayout)
            patientDict['pane'] = patientPane

            # Make patient vitals data accessable by other functions
            self.vitals.append(patientDict)

            if (i != 0):
                patientPane.setVisible(False)

            # Add this patient to the overall vitals pane
            vitalsPaneLayout.addWidget(patientPane,i,0)
        
        self.vitalsPane.setLayout(vitalsPaneLayout)


    def initColorGradient(self):
        self.colorGradient = pg.GradientWidget(orientation='right')
        self.colorGradient.restoreState(self.gradientMode)
        self.colorGradient.setVisible(False)


    def init3dGraph(self):
        # Create plot
        self.pcplot = gl.GLViewWidget()
        # Sets background to a pastel grey
        self.pcplot.setBackgroundColor(70, 72, 79)
        # Create the background grid
        self.gz = gl.GLGridItem()
        self.pcplot.addItem(self.gz)

        # Create scatter plot for point cloud
        self.scatter = gl.GLScatterPlotItem(size=5)
        self.scatter.setData(pos=np.zeros((1,3)))
        self.pcplot.addItem(self.scatter)
        
        # Create box to represent EVM
        evmSizeX = 0.0625
        evmSizeZ = 0.125
        verts = np.empty((2,3,3))
        verts[0,0,:] = [-evmSizeX, 0, evmSizeZ]
        verts[0,1,:] = [-evmSizeX,0,-evmSizeZ]
        verts[0,2,:] = [evmSizeX,0,-evmSizeZ]
        verts[1,0,:] = [-evmSizeX, 0, evmSizeZ]
        verts[1,1,:] = [evmSizeX, 0, evmSizeZ]
        verts[1,2,:] = [evmSizeX, 0, -evmSizeZ]
        self.evmBox = gl.GLMeshItem(vertexes=verts,smooth=False,drawEdges=True,edgeColor=pg.glColor('r'),drawFaces=False)
        self.pcplot.addItem(self.evmBox)

        # Initialize other elements
        self.boundaryBoxViz = []
        self.coordStr = []
        self.ellipsoids = []

        
    def updateGraph(self, outputDict):
        pointCloud = None
        numPoints = 0
        classifierOutput = None
        tracks = None
        trackIndexs = None
        numTracks = 0
        self.frameNum = 0
        error = 0
        occupancyStates = None
        vitalsDict = None
        self.useFilter = 0
        heights = None
        
        # Point Cloud
        if ('pointCloud' in outputDict):
            pointCloud = outputDict['pointCloud']
        
        # Number of Points
        if ('numDetectedPoints' in outputDict):
            numPoints = outputDict['numDetectedPoints']

        # Tracks
        if ('trackData' in outputDict):
            tracks = outputDict['trackData']

        # Heights
        if ('heightData' in outputDict):
            heights = outputDict['heightData']
    
        # Track index
        if ('trackIndexes' in outputDict):
            trackIndexs = outputDict['trackIndexes']
            
        # Number of Tracks
        if ('numDetectedTracks' in outputDict):
            numTracks = outputDict['numDetectedTracks']

        # Frame number
        if ('frameNum' in outputDict):
            self.frameNum = outputDict['frameNum'] 

        # Error
        if ('error' in outputDict):
            error = outputDict['error']
            
        # Occupancy State Machine
        if ('occupancy' in outputDict):
            occupancyStates = outputDict['occupancy']

        # Vital Signs Info
        if ('vitals' in outputDict):
            vitalsDict = outputDict['vitals']
            
        if (error != 0):
            print ("Parsing Error on frame: %d" % (self.frameNum))
            print ("\tError Number: %d" % (error))
        
        # Update text for display
        self.numPointsDisplay.setText('Points: '+str(numPoints))
        self.numTargetsDisplay.setText('Targets: '+str(numTracks))

        # Rotate point cloud and tracks to account for elevation and azimuth tilt
        if (self.profile['elev_tilt'] != 0 or self.profile['az_tilt'] != 0):
            if (pointCloud is not None):
                for i in range(numPoints):
                    rotX, rotY, rotZ = eulerRot (pointCloud[i,0], pointCloud[i,1], pointCloud[i,2], self.profile['elev_tilt'], self.profile['az_tilt'])
                    pointCloud[i,0] = rotX
                    pointCloud[i,1] = rotY
                    pointCloud[i,2] = rotZ
            if (tracks is not None):
                for i in range(numTracks):
                    rotX, rotY, rotZ = eulerRot (tracks[i,1], tracks[i,2], tracks[i,3], self.profile['elev_tilt'], self.profile['az_tilt'])
                    tracks[i,1] = rotX
                    tracks[i,2] = rotY
                    tracks[i,3] = rotZ

        # Shift points to account for sensor height
        if (self.profile['sensorHeight'] != 0):
            if (pointCloud is not None):
                pointCloud[:,2] = pointCloud[:,2] + self.profile['sensorHeight']
            if (tracks is not None):
                tracks[:,3] = tracks[:,3] + self.profile['sensorHeight']

        # Update boundary box colors based on results of Occupancy State Machine
        if (occupancyStates is not None):
            for box in self.boundaryBoxes:
                if ('occZone' in box['name']):
                    # Get index of the occupancy zone from the box name
                    occIdx = int(box['name'].lstrip(string.ascii_letters))
                    # Zone unnoccupied 
                    if (occIdx >= len(occupancyStates) or not occupancyStates[occIdx]):
                        box['color'].setCurrentText('Green')
                    # Zone occupied
                    else:
                        # Make first box turn red
                        if (occIdx == 0):
                            box['color'].setCurrentText('Red')
                        else:
                            box['color'].setCurrentText('Yellow')

        # Vital Signs info
        if (vitalsDict is not None):
            # Update info for each patient
            patientId = vitalsDict['id']
            # Check that patient id is valid
            if (patientId < self.profile['maxTracks']):
                self.vitalsPatientData[patientId]['rangeBin'] = vitalsDict['rangeBin']
                self.vitalsPatientData[patientId]['breathDeviation'] = vitalsDict['breathDeviation']
                self.vitalsPatientData[patientId]['breathRate'] = vitalsDict['breathRate']

                # Take the median of the last n heartrates to prevent it from being sporadic
                self.vitalsPatientData[patientId]['heartRate'].append(vitalsDict['heartRate'])
                while (len(self.vitalsPatientData[patientId]['heartRate']) > NUM_HEART_RATES_FOR_MEDIAN):
                    self.vitalsPatientData[patientId]['heartRate'].pop(0)
                medianHeartRate = statistics.median(self.vitalsPatientData[patientId]['heartRate'])
                
                # Check if the patient is holding their breath, and if there is a patient  detected at all
                # TODO ensure vitals output is 0 
                if(float(vitalsDict['breathDeviation']) == 0 or numTracks == 0):
                    patientStatus = 'No Patient Detected'
                    breathRateText = "N/A"
                    heartRateText = "N/A"
                    # Workaround to ensure waveform is flat when no track is present
                    for i in range(NUM_FRAMES_PER_VITALS_PACKET):
                        vitalsDict['heartWaveform'][i] = 0
                        vitalsDict['breathWaveform'][i] = 0
                else:
                    heartRateText = str(round(medianHeartRate, 1))
                    # Patient breathing normally
                    if (float(vitalsDict['breathDeviation']) >= 0.02):
                        patientStatus = 'Presence'
                        # Round the floats to 1 decimal place and format them for display
                        breathRateText = str(round(self.vitalsPatientData[patientId]['breathRate'], 1))
                     # Patient holding breath
                    else:
                        patientStatus = 'Holding Breath'
                        breathRateText = "N/A"
                 
                # Add heart rate waveform data for this packet to the graph
                self.vitalsPatientData[patientId]['heartWaveform'].extend(vitalsDict['heartWaveform'])
                while (len(self.vitalsPatientData[patientId]['heartWaveform']) > NUM_VITALS_FRAMES_IN_PLOT):
                    self.vitalsPatientData[patientId]['heartWaveform'].pop(0)

                # Add breathing rate waveform data for this packet to the graph
                self.vitalsPatientData[patientId]['breathWaveform'].extend(vitalsDict['breathWaveform'])
                while (len(self.vitalsPatientData[patientId]['breathWaveform']) > NUM_VITALS_FRAMES_IN_PLOT):
                    self.vitalsPatientData[patientId]['breathWaveform'].pop(0)

                # Copy waveforms so that we can reverse their oritentation
                heartWaveform = self.vitalsPatientData[patientId]['heartWaveform'].copy()
                heartWaveform.reverse()

                # Copy waveforms so that we can reverse their oritentation
                breathWaveform = self.vitalsPatientData[patientId]['breathWaveform'].copy()
                breathWaveform.reverse()

                # Update relevant info in GUI
                self.vitals[patientId]['heartGraph'].setData(heartWaveform)
                self.vitals[patientId]['breathGraph'].setData( breathWaveform)
                self.vitals[patientId]['heartRate'].setText(heartRateText)
                self.vitals[patientId]['breathRate'].setText(breathRateText)
                self.vitals[patientId]['status'].setText(patientStatus)
                self.vitals[patientId]['rangeBin'].setText(str(self.vitalsPatientData[patientId]['rangeBin']))

        #reset all heights each loop to delete heights from tracks that disappear.
        for cstr in self.coordStr:
            cstr.setVisible(False)

        #target heights        
        if (heights is not None):
            if (len(heights) != len(tracks)):
                print ("WARNING: number of heights does not match number of tracks")
            #populate heights for current tracks
            for height in heights:
                # Find track with correct TID
                for track in tracks:
                    # Found correct track
                    if (track[0] == height[0]):
                        tid = int(height[0])
                        height_str = 'tid : ' + str(height[0]) + ', height : ' + str(round(height[1], 2)) + ' m'
                        self.coordStr[tid].setText(height_str)
                        self.coordStr[tid].setX(track[1])
                        self.coordStr[tid].setY(track[2])
                        self.coordStr[tid].setZ(track[3])
                        self.coordStr[tid].setVisible(True)
                        break

        # Point cloud Persistence
        numPersistentFrames = int(self.persistentFramesInput.currentText())
        if (self.configType.currentText() == DEMO_NAME_3DPC or self.configType.currentText() == DEMO_NAME_VITALS):
            numPersistentFrames = numPersistentFrames + 1

        # Add trackIndexs to the point cloud before adding it to the cumulative cloud
        if (trackIndexs is not None):
            if (self.configType.currentText() == DEMO_NAME_3DPC or self.configType.currentText() == DEMO_NAME_VITALS):
                if (self.previousClouds[len(self.previousClouds) - 1].shape[0] != trackIndexs.shape[0]):
                    print ("Error in gui_main.py: number of points in last frame (" + str(self.previousClouds[len(self.previousClouds) - 1].shape[0]) + ") does not match number of track indexes (" + str(trackIndexs.shape[0])+ ")")
                else:
                    self.previousClouds[len(self.previousClouds) - 1][:, 6] = trackIndexs
            else:
                if (pointCloud.shape[0] != trackIndexs.shape[0]):
                    print ("Error in gui_main.py: number of points does not match number of track indexes")
                else:
                    pointCloud[:, 6] = trackIndexs

        # Add current point cloud to the cumulative cloud
        self.previousClouds.append(pointCloud)
        # If we have more point clouds than we need, delete the oldest ones
        while (len(self.previousClouds) > numPersistentFrames):
            self.previousClouds.pop(0)
            
        # Since track indexes are delayed a frame, delay showing the current points by 1 frame
        if (self.frameNum > 1 and (self.configType.currentText() == DEMO_NAME_3DPC or self.configType.currentText() == DEMO_NAME_VITALS)):
            cumulativeCloud = np.concatenate(self.previousClouds[:-1])
        else:
            cumulativeCloud = np.concatenate(self.previousClouds)

        # Update graph, but first ensure the last update completed
        if (self.graphFin):
            self.plotstart = int(round(time.time()*1000))
            self.graphFin = 0
            self.get_thread = updateQTTargetThread3D(cumulativeCloud, tracks, self.scatter, self.pcplot, numTracks, self.ellipsoids, self.coordStr, classifierOutput, self.zRange, self.initColorGradient, self.pointColorMode.currentText(), self.plotTracks.isChecked(), self.trackColorMap)
            self.get_thread.done.connect(self.graphDone)
            self.get_thread.start(priority=QThread.HighestPriority-1)


    def graphDone(self):
        plotend = int(round(time.time()*1000))
        plotime = plotend - self.plotstart
        try:
            if (self.frameNum > 1):
                self.averagePlot = (plotime*1/self.frameNum) + (self.averagePlot*(self.frameNum-1)/(self.frameNum))
            else:
                self.averagePlot = plotime
        except:
            self.averagePlot = plotime
        self.graphFin = 1
        pltstr = 'Average Plot time: '+str(plotime)[:5] + ' ms'
        fnstr = 'Frame: '+str(self.frameNum)
        self.frameNumDisplay.setText(fnstr)
        self.plotTimeDisplay.setText(pltstr)


    def resetFallText(self):
        self.fallAlert.setText('Standing')
        self.fallPic.setPixmap(self.standingPicture)
        self.fallResetTimerOn = 0


    def updateFallThresh(self):
        try:
            newThresh = float(self.fallThreshInput.text())
            self.fallThresh = newThresh
            self.fallThreshMarker.setPos(self.fallThresh)
        except:
            print('No numberical threshold')


    def connectCom(self):
        #get parser
        self.parser = uartParser(type=self.configType.currentText())
        self.parser.frameTime = self.frameTime
        print('Parser type: ',self.configType.currentText())
        #init threads and timers
        self.uart_thread = parseUartThread(self.parser)
        if (self.configType.currentText() != 'Replay'):
            self.uart_thread.fin.connect(self.parseData)
        self.uart_thread.fin.connect(self.updateGraph)
        self.parseTimer = QTimer()
        self.parseTimer.setSingleShot(False)
        self.parseTimer.timeout.connect(self.parseData)        
        try:
            uart = "COM"+ self.cliCom.text()
            data = "COM"+ self.dataCom.text()
            self.parser.connectComPorts(uart, data)
            self.connectStatus.setText('Connected')
            self.connectButton.setText('Disconnect')
#TODO: create the disconnect button action
        except Exception as e:
            print (e)
            self.connectStatus.setText('Unable to Connect')
        if (self.configType.currentText() == "Replay"):
            self.connectStatus.setText('Replay')
        if (self.configType.currentText() == DEMO_NAME_LRPD):
            self.frameTime = 400
#
# Select and parse the configuration file
# TODO select the cfgfile automatically based on the profile.
    def selectCfg(self):
        try:
            self.parseCfg(self.selectFile())
        except Exception as e:
            print(e)
            print('No cfg file selected!')
    
    def selectFile(self):
        try:

            current_dir = os.getcwd()
            configDirectory = current_dir
        except:
            configDirectory = ''
        
        fd = QFileDialog()
        filt = "cfg(*.cfg)"
        filename = fd.getOpenFileName(directory=configDirectory,filter=filt)
        return filename[0]
    

    # Add a boundary box to the boundary boxes tab
    def addBoundBox(self, name, minX=0, maxX=0, minY=0, maxY=0, minZ=0, maxZ=0):
        newBox = self.setBoxControlLayout(name)
        self.boundaryBoxes.append(newBox)
        self.boundaryBoxViz.append(gl.GLLinePlotItem())
        boxIndex = len(self.boundaryBoxes) - 1
        self.boxTab.addTab(newBox['boxCon'], name)
        self.boundaryBoxes[boxIndex]['boundList'][0].setText(str(minX))
        self.boundaryBoxes[boxIndex]['boundList'][1].setText(str(maxX))
        self.boundaryBoxes[boxIndex]['boundList'][2].setText(str(minY))
        self.boundaryBoxes[boxIndex]['boundList'][3].setText(str(maxY))
        self.boundaryBoxes[boxIndex]['boundList'][4].setText(str(minZ))
        self.boundaryBoxes[boxIndex]['boundList'][5].setText(str(maxZ))
        
        # Specific functionality for various types of boxes
        # Point boundary box
        if ('pointBounds' in name):
            desc = 'Throw away points outside of this zone'
            self.boundaryBoxes[boxIndex]['description'].setText(desc)
            self.boundaryBoxes[boxIndex]['checkEnable'].setDisabled(True) # Lock ability to enable this since its not implemented
        # Zone occupancy box
        elif ('occZone' in name):
            desc = 'Checks occupancy status on these zones'
            self.boundaryBoxes[boxIndex]['description'].setText(desc)
            self.boundaryBoxes[boxIndex]['checkEnable'].setChecked(True)
            self.boundaryBoxes[boxIndex]['color'].setCurrentText('Green')
            # Lock each text field
            for textBox in self.boundaryBoxes[boxIndex]['boundList']:
                textBox.setDisabled(True)
            # Lock enable box
            self.boundaryBoxes[boxIndex]['checkEnable'].setDisabled(True)
            self.boundaryBoxes[boxIndex]['color'].setDisabled(True)
        elif ('trackerBounds' in name):
            desc = 'Checks for tracks in this zone'
            self.boundaryBoxes[boxIndex]['description'].setText(desc)
            self.boundaryBoxes[boxIndex]['checkEnable'].setChecked(True)
            # Lock each text field
            for textBox in self.boundaryBoxes[boxIndex]['boundList']:
                textBox.setDisabled(True)
            # Lock enable box
            self.boundaryBoxes[boxIndex]['checkEnable'].setDisabled(True)

        # Set visible if enabled
        if (self.boundaryBoxes[boxIndex]['checkEnable'].isChecked()):
            self.boundaryBoxViz[boxIndex].setVisible(True)
        else:
            self.boundaryBoxViz[boxIndex].setVisible(False)
        self.pcplot.addItem(self.boundaryBoxViz[boxIndex])
        self.onChangeBoundaryBox()
        

    def parseCfg(self, fname):
        with open(fname, 'r') as cfg_file:
            self.cfg = cfg_file.readlines()
        counter = 0
        chirpCount = 0
        for line in self.cfg:
            args = line.split()
            if (len(args) > 0):
                # cfarCfg
                if (args[0] == 'cfarCfg'):
                    pass
                    #self.cfarConfig = {args[10], args[11], '1'}
                # trackingCfg
                elif (args[0] == 'trackingCfg'):
                    if (len(args) < 5):
                        print ("Error: trackingCfg had fewer arguments than expected")
                        continue
                    self.profile['maxTracks'] = int(args[4])
                    self.trackColorMap = get_trackColors(self.profile['maxTracks'])
                    for m in range(self.profile['maxTracks']):
                        # Add track gui object
                        mesh = gl.GLLinePlotItem()
                        mesh.setVisible(False)
                        self.pcplot.addItem(mesh)
                        self.ellipsoids.append(mesh)
                         #add track coordinate string
                        text = GLTextItem()
                        text.setGLViewWidget(self.pcplot)
                        text.setVisible(False)
                        self.pcplot.addItem(text)
                        self.coordStr.append(text)
                    # If we only support 1 patient, hide the other patient window
                    if (self.profile['maxTracks'] == 1):
                        self.vitals[1]['pane'].setVisible(False)
                    # Initialize Vitals output dictionaries for each potential patient
                    for i in range (min(self.profile['maxTracks'], MAX_VITALS_PATIENTS)):
                        # Initialize 
                        patientDict = {}
                        patientDict ['id'] = i
                        patientDict ['rangeBin'] = 0
                        patientDict ['breathDeviation'] = 0
                        patientDict ['heartRate'] = []
                        patientDict ['breathRate'] = 0
                        patientDict ['heartWaveform'] = []
                        patientDict ['breathWaveform'] = []
                        self.vitalsPatientData.append(patientDict)

                        # Make each patient's pane visible
                        self.vitals[i]['pane'].setVisible(True)

                elif (args[0] == 'AllocationParam'):
                    pass
                    #self.allocConfig = tuple(args[1:6])
                elif (args[0] == 'GatingParam'):
                    pass
                    #self.gatingConfig = tuple(args[1:4])
                elif (args[0] == 'SceneryParam' or args[0] == 'boundaryBox'):
                    if (len(args) < 7):
                        print ("Error: SceneryParam/boundaryBox had fewer arguments than expected")
                        continue
                    self.boundaryLine = counter
                    leftX = float(args[1])
                    rightX = float(args[2])
                    nearY = float(args[3])
                    farY = float(args[4])
                    bottomZ = float(args[5])
                    topZ = float(args[6])
                    self.addBoundBox('trackerBounds', leftX, rightX, nearY, farY, bottomZ, topZ)
                elif (args[0] == 'staticBoundaryBox'):
                    self.staticLine = counter
                elif (args[0] == 'profileCfg'):
                    if (len(args) < 12):
                        print ("Error: profileCfg had fewer arguments than expected")
                        continue
                    self.profile['startFreq'] = float(args[2])
                    self.profile['idle'] = float(args[3])
                    self.profile['adcStart'] = float(args[4])
                    self.profile['rampEnd'] = float(args[5])
                    self.profile['slope'] = float(args[8])
                    self.profile['samples'] = float(args[10])
                    self.profile['sampleRate'] = float(args[11])
                    print(self.profile)
                elif (args[0] == 'frameCfg'):
                    if (len(args) < 4):
                        print ("Error: frameCfg had fewer arguments than expected")
                        continue
                    self.profile['numLoops'] = float(args[3])
                    self.profile['numTx'] = float(args[2])+1
                elif (args[0] == 'chirpCfg'):
                    chirpCount += 1
                elif (args[0] == 'sensorPosition'):
                    if (len(args) < 4):
                        print ("Error: sensorPosition had fewer arguments than expected")
                        continue
                    self.profile['sensorHeight'] = float(args[1])
                    print('Sensor Height from cfg = ',str(self.profile['sensorHeight']))
                    self.profile['az_tilt'] = float(args[2])
                    self.profile['elev_tilt'] = float(args[3])
                 # Only used for Small Obstacle Detection
                elif (args[0] == 'occStateMach'):
                    numZones = int(args[1])
                    if (numZones > 2):
                        print('ERROR: More zones specified by cfg than are supported in this GUI')
                # Only used for Small Obstacle Detection
                elif (args[0] == 'zoneDef'):
                    if (len(args) < 8):
                        print ("Error: zoneDef had fewer arguments than expected")
                        continue
                    zoneIdx = int(args[1])
                    minX = float(args[2])
                    maxX = float(args[3])
                    minY = float(args[4])
                    maxY = float(args[5])
                    # Offset by 3 so it is in center of screen
                    minZ = float(args[6]) + self.profile['sensorHeight']
                    maxZ = float(args[7]) + self.profile['sensorHeight']

                    name = 'occZone' + str(zoneIdx)

                    self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)
                
            counter += 1

        self.profile['maxRange'] = self.profile['sampleRate']*1e3*0.9*3e8/(2*self.profile['slope']*1e12)
        bw = self.profile['samples']/(self.profile['sampleRate']*1e3)*self.profile['slope']*1e12
        rangeRes = 3e8/(2*bw)
        Tc = (self.profile['idle']*1e-6 + self.profile['rampEnd']*1e-6)*chirpCount
        lda = 3e8/(self.profile['startFreq']*1e9)
        maxVelocity = lda/(4*Tc)
        velocityRes = lda/(2*Tc*self.profile['numLoops']*self.profile['numTx'])
        self.configTable.setItem(1,1,QTableWidgetItem(str(self.profile['maxRange'])[:5]))
        self.configTable.setItem(2,1,QTableWidgetItem(str(rangeRes)[:5]))
        self.configTable.setItem(3,1,QTableWidgetItem(str(maxVelocity)[:5]))
        self.configTable.setItem(4,1,QTableWidgetItem(str(velocityRes)[:5]))
        
        # Update sensor position
        self.az_tilt.setText(str(self.profile['az_tilt']))
        self.elev_tilt.setText(str(self.profile['elev_tilt']))
        self.s_height.setText(str(self.profile['sensorHeight']))
        self.onChangeSensorPosition()

    def sendCfg(self):
        try:
            if (self.configType.currentText() != "Replay"):
                self.parser.sendCfg(self.cfg)
                self.configSent = 1
                self.parseTimer.start(self.frameTime) # need this line 
                
        except Exception as e:
            print(e)
            print ('No cfg file selected!')

    def startApp(self):
        self.configSent = 1
        self.parseTimer.start(self.frameTime) # need this line 

    def parseData(self):
        self.uart_thread.start(priority=QThread.HighestPriority)

    def whoVisible(self):
        if (self.threeD):
            self.threeD = 0
        else:
            self.threeD = 1
        print('3d: ', self.threeD)

if __name__ == '__main__':
    if (compileGui):
        appctxt = ApplicationContext()
        app = QApplication(sys.argv)
        screen = app.primaryScreen()
        size = screen.size()
        main = Window(size=size)
        main.show()
        exit_code = appctxt.app.exec_()
        sys.exit(exit_code)
    else:
        app = QApplication(sys.argv)
        screen = app.primaryScreen()
        size = screen.size()
        main = Window(size=size)
        main.show()
        sys.exit(app.exec_())
