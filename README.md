# WEC Graph Tools

Two Colab notebooks for Houston CSV logs. Data lives on Drive: `WEC_Electronics/Data`
(https://drive.google.com/drive/folders/115T7YsGx1EHJLYIa-piAbEpjV60-vW_y).

| Notebook | Code | What it does |
|---|---|---|
| `GraphGeneral.ipynb` | `graphutils.py` | Plot columns vs time, binned averages, 3D scatter, CWC score |
| `PIDTuning.ipynb` | `pid_tuning.py` | Find PID gains for constant power / constant resistance mode |

Open a notebook in Colab, run **Setup** once, then edit the file/column names and run the cells.
Setup downloads the `.py` file from this repo's `main` branch.

## Graphs (`graphutils.py`)

```python
df = load_data("testing-5-13/tunnel_testing_01.csv")  # path relative to WEC_Electronics/Data
list(df.columns)                                      # exact column names in this file
graph_2d(df, "Load Power", "Set Power")               # any number of columns
graph_2d(df, "Load Power", timestart=30, timeend=60)  # zoom, seconds
graph_2d_bar(df, "Load Power", resample=2)            # 2 s averages
graph_3d(df, "Time", "Load Power", "Load Voltage")    # interactive 3D
final_score(df)                                       # needs a "Wind Speed" column
```

Time is always seconds from the start of the file. Column names must match exactly;
a wrong name prints the real ones. The DAC column was renamed over time:
`DAC Output` (to Apr 2026) -> `Output Voltage` (to Sep 2026) -> `DAC Output Voltage`.

## PID tuning (`pid_tuning.py`)

1. On the bench, log a run where the set point steps (e.g. Set Power 5 -> 6 -> 4 -> 5, 10 s each).
2. In `PIDTuning.ipynb` set `FILES`, `Y_COL` (`'Load Power'` or `'Load Resistance'`), `SETPOINT`,
   and `START`/`END` in seconds (skip the part near 0 W/0 A), then run **Tune**.
3. Type the printed values into Houston Live Data (`Pow Kp`, `Pow Kd`, ... ) or paste them into `main.c`.

Command line works too:

```bash
python pid_tuning.py run.csv --y "Load Power" --setpoint 5 --start 20
python pid_tuning.py a.csv b.csv --y "Load Resistance" --setpoint 25   # one gain set for several runs
python pid_tuning.py                                                  # self-check
```

Both files run a self-check with `python graphutils.py` / `python pid_tuning.py`.
