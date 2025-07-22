# ----------------------------------------------
# Script Recorded by Ansys Electronics Desktop Version 2024.2.0
# 16:04:53  Jul 21, 2025
# ----------------------------------------------
import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.SetActiveProject("PeetsFEAdev_Project")
oDesign = oProject.SetActiveDesign("PeetsFEAdev_Design")
oModule = oDesign.GetModule("ReportSetup")
oModule.CreateReport("copper loss data", "Fields", "Data Table", "Setup1 : LastAdaptive", [],
                     [
    "Freq:=", ["All"],
    "Phase:=", ["0deg"],
    "w1:=", ["Nominal"],
    "l1_leg:=", ["Nominal"],
    "l1_top:=", ["Nominal"],
    "l1_center:=", ["Nominal"],
    "l2:=", ["Nominal"],
    "h1:=", ["Nominal"],
    "l2_tap:=", ["Nominal"],
    "ratio:=", ["Nominal"],
    "Tx_turns:=", ["Nominal"],
    "Tx_tap:=", ["Nominal"],
    "Tx_height:=", ["Nominal"],
    "Tx_preg:=", ["Nominal"],
    "Rx_turns:=", ["Nominal"],
    "Rx_tap:=", ["Nominal"],
    "Rx_height:=", ["Nominal"],
    "Rx_preg:=", ["Nominal"],
    "g1:=", ["Nominal"],
    "g2:=", ["Nominal"],
    "Tx_space_x:=", ["Nominal"],
    "Tx_space_y:=", ["Nominal"],
    "Rx_space_x:=", ["Nominal"],
    "Rx_space_y:=", ["Nominal"],
    "core_N_w1:=", ["Nominal"],
    "core_P_w1:=", ["Nominal"],
    "Tx_layer_space_x:=", ["Nominal"],
    "Tx_layer_space_y:=", ["Nominal"],
    "Rx_layer_space_x:=", ["Nominal"],
    "Rx_layer_space_y:=", ["Nominal"],
    "Tx_width:=", ["Nominal"],
    "Rx_width:=", ["Nominal"],
    "seed:=", ["Nominal"],
    "Tx_hTurns:=", ["Nominal"],
    "Rx_hTurns:=", ["Nominal"]
],
    [
    "X Component:=", "Freq",
    "Y Component:=", ["P_Tx_1", "P_Rx_False_True_1", "P_Rx_True_True_1"]
])
