# Wind Turbine Experimental Data Visualization Tool

This notebook is designed to easily visualize various experimental data (resistance, power, voltage, current, etc.).

## How to Use
0. The Dataset should be uploaded here:
https://drive.google.com/drive/folders/115T7YsGx1EHJLYIa-piAbEpjV60-vW_y
1. Run the **[Setup & Load Data]** cell first. (Wait for a new google login window to pop up. If it doesn't, stop the code and restart it)
2. Go to the **[2D Graph]** or **[3D Graph]** cells, change the variable names, and run them.
    * **2D Graph:** `graph_2d("filename.csv", "Variable 1", "Variable 2", ...)`
   * **3D Graph:** `graph_3d("filename.csv", "X-axis Variable", "Y-axis Variable", "Z-axis Variable")`
    * If your file is inside certain folder, for example **"test4.csv"** file in folder named **"testing_2-15-2026"** which should be under ".../WEC_Electronics/Data" then file name should be **"testing_2-15-2026/test4.csv"**.

## Available Variables (Column Names)
Copy and paste the names exactly as written below:
* `Time`
* `Set Resistance`
* `Set Voltage`
* `Set Current`
* `Set Power`
* `Load Resistance`
* `Load Voltage`
* `Load Current`
* `Load Power`
* `DAC Output`
* `Servo Position`
* `Wind Speed`
