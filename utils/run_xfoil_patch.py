"""
This module implements a patch to the aerosandbox method aerosandbox.aerodynamics.aero_2D.xfoil.Xfoil.run_xfoil to let code run despite header size mismatch between asb expected and xfoil output 

patch is applied only once 
"""

import inspect
import os
from pathlib import Path
import re
import subprocess
import tempfile
import warnings

import numpy as np
import aerosandbox.aerodynamics.aero_2D.xfoil as xfoil_module


if not getattr(xfoil_module.XFoil, "_local_patch_applied", False):

    source = inspect.getsource(xfoil_module.XFoil._run_xfoil)

    if "header and data have different numbers of columns" not in source:
        raise RuntimeError(
            "AeroSandbox's XFoil parser has changed. "
            "Review this patch before applying it."
        )

    def patched_run_xfoil(self, run_command, read_bl_data_from=None):
        """Run XFoil. (Private function.)"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            if self.working_directory is not None:
                directory = Path(self.working_directory)

            output_filename = "output.txt"
            airfoil_file = "airfoil.dat"
            self.airfoil.write_dat(directory / airfoil_file)

            keystrokes = self._default_keystrokes(
                airfoil_filename=airfoil_file, output_filename=output_filename
            )
            keystrokes += [run_command]
            keystrokes += ["pacc", "", "quit"]

            try:
                os.remove(directory / output_filename)
            except FileNotFoundError:
                pass

            try:
                proc = subprocess.Popen(
                    self.xfoil_command,
                    cwd=directory,
                    stdin=subprocess.PIPE,
                    stdout=None if self.verbose else subprocess.DEVNULL,
                    stderr=None if self.verbose else subprocess.DEVNULL,
                    text=True,
                )
            except FileNotFoundError as e:
                raise self.XFoilError(
                    f"Could not launch XFoil with the command '{self.xfoil_command}'.\n"
                    "This is likely because AeroSandbox does not see XFoil on PATH with the given command.\n"
                    "Provide the correct path to the XFoil executable in the asb.XFoil constructor via `xfoil_command=`."
                ) from e

            try:
                outs, errs = proc.communicate(
                    input="\n".join(keystrokes), timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()

                warnings.warn(
                    "XFoil run timed out!\n"
                    "If this was not expected, try increasing the `timeout` parameter\n"
                    "when you create this AeroSandbox XFoil instance.",
                    stacklevel=2,
                )

            return_code = proc.returncode

            if return_code in (11, -11, 139, 3221225477):
                raise self.XFoilError(
                    "XFoil segmentation-faulted. This is likely because your input airfoil has too many points.\n"
                    "Try repaneling your airfoil with `Airfoil.repanel()` before passing it into XFoil.\n"
                    "For further debugging, turn on the `verbose` flag when creating this AeroSandbox XFoil instance."
                )
            elif return_code in (8, -8, 136):
                raise self.XFoilError(
                    "XFoil returned a floating point exception. This is probably because you are trying to start\n"
                    "your analysis at an operating point where the viscous boundary layer can't be initialized based\n"
                    "on the computed inviscid flow. (You're probably hitting a Goldstein singularity.) Try starting\n"
                    "your XFoil run at a less-aggressive (alpha closer to 0, higher Re) operating point."
                )

            try:
                with open(directory / output_filename) as f:
                    lines = f.readlines()
            except FileNotFoundError:
                raise self.XFoilError(
                    "It appears XFoil didn't produce an output file, probably because it crashed.\n"
                    "To troubleshoot, try some combination of the following:\n"
                    "\t - In the XFoil constructor, verify that either XFoil is on PATH or that the `xfoil_command` parameter is set.\n"
                    "\t - In the XFoil constructor, run with `verbose=True`.\n"
                    "\t - In the XFoil constructor, set the `working_directory` parameter to a known folder to see the XFoil input and output files.\n"
                    "\t - In the XFoil constructor, set the `timeout` parameter to a large number to see if XFoil is just taking a long time to run.\n"
                    "\t - On Windows, use `XFoil.open_interactive()` to run XFoil interactively in a new window.\n"
                    "\t - Try allowing XFoil to repanel the airfoil by setting `xfoil_repanel=True` in the XFoil constructor."
                )

            separator_line = None
            for i, line in enumerate(lines):
                if line.count("-") >= 30:
                    separator_line = i
                    break

            if separator_line is None:
                raise self.XFoilError(
                    "XFoil output file is malformed; it doesn't have the expected number of lines.\n"
                    "For debugging, the raw output file from XFoil is printed below:\n"
                    + "\n".join(lines)
                )

            title_line = lines[separator_line - 1]
            columns = title_line.split()
            data_lines = lines[separator_line + 1 :]

            def str_to_float(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return np.nan

            output = {
                column: []
                for column in [
                    "alpha",
                    "CL",
                    "CD",
                    "CDp",
                    "CM",
                    "Cpmin",
                    "Xcpmin",
                    "Chinge",
                    "Top_Xtr",
                    "Bot_Xtr",
                ]
            }

            for pointno, line in enumerate(data_lines):
                float_pattern = r"-?\d+\.\d+"
                entries = re.findall(float_pattern, line)
                data = [str_to_float(entry) for entry in entries]

                if len(data) == 10 and len(columns) == 8:
                    columns = [
                        "alpha",
                        "CL",
                        "CD",
                        "CDp",
                        "CM",
                        "Cpmin",
                        "Xcpmin",
                        "Chinge",
                        "Top_Xtr",
                        "Bot_Xtr",
                    ]

                if len(data) > len(columns):
                    data = data[: len(columns)]
                elif len(data) < len(columns):
                    raise self.XFoilError(
                        "XFoil output file is malformed; the header and data have different numbers of columns.\n"
                        "In previous testing, this occurs due to a bug in XFoil itself, with certain input combos.\n"
                        "For debugging, the raw output file from XFoil is printed below:\n"
                        + "\n".join(lines)
                        + "\nTitle line: "
                        + title_line
                        + f"\nIdentified {len(data)} data columns and {len(columns)} header columns."
                        + "\nColumns: "
                        + str(columns)
                        + "\nData: "
                        + str(data)
                    )

                for i in range(len(columns)):
                    if columns[i] in output:
                        output[columns[i]].append(data[i])

            output = {k: np.array(v, dtype=float) for k, v in output.items()}

            if read_bl_data_from is not None:
                import pandas as pd

                bl_datas: list[pd.DataFrame] = []

                if read_bl_data_from == "alpha":
                    alpha_to_dump_mapping = {
                        float(dump_filename.stem.split("_")[-1]): dump_filename
                        for dump_filename in directory.glob("dump_a_*.txt")
                    }

                    for alpha in output["alpha"]:
                        dump_filename = alpha_to_dump_mapping[
                            min(
                                alpha_to_dump_mapping.keys(),
                                key=lambda x: abs(x - alpha),
                            )
                        ]

                        bl_datas.append(
                            pd.read_csv(
                                dump_filename,
                                sep=r"\s+",
                                names=[
                                    "s",
                                    "x",
                                    "y",
                                    "ue/vinf",
                                    "dstar",
                                    "theta",
                                    "cf",
                                    "H",
                                ],
                                skiprows=1,
                            )
                        )

                elif read_bl_data_from == "cl":
                    cl_to_dump_mapping = {
                        float(dump_filename.stem.split("_")[-1]): dump_filename
                        for dump_filename in directory.glob("dump_cl_*.txt")
                    }

                    for cl in output["CL"]:
                        dump_filename = cl_to_dump_mapping[
                            min(cl_to_dump_mapping.keys(), key=lambda x: abs(x - cl))
                        ]

                        bl_datas.append(
                            pd.read_csv(
                                dump_filename,
                                sep=r"\s+",
                                names=[
                                    "s",
                                    "x",
                                    "y",
                                    "ue/vinf",
                                    "dstar",
                                    "theta",
                                    "cf",
                                    "H",
                                ],
                                skiprows=1,
                            )
                        )

                else:
                    raise ValueError(
                        "The `read_bl_data_from` parameter must be 'alpha', 'cl', or None."
                    )

                for bl_data in bl_datas:
                    Cp_0 = 1 - bl_data["ue/vinf"] ** 2
                    bl_data["Cp"] = Cp_0 / (
                        np.sqrt(1 - self.mach**2)
                        + (
                            (self.mach**2)
                            / (1 + np.sqrt(1 - self.mach**2))
                            * (Cp_0 / 2)
                        )
                    )

                    bl_data["Re_theta"] = (
                        np.abs(bl_data["ue/vinf"]) * bl_data["theta"] * self.Re
                    )

                output["bl_data"] = np.fromiter(bl_datas, dtype="O")

            return output
    
    xfoil_module.XFoil._run_xfoil = patched_run_xfoil
    xfoil_module.XFoil._local_patch_applied = True
