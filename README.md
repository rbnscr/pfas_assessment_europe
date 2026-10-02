# PFAS assessment in European surface waters

[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--9155--2753-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-9155-2753)

Code for: *Fingerprinting PFAS pollution in European surface waters*

The version associated with the manuscript is archived on Zenodo: 

This repository contains the analysis code and supporting information for the accompanying research paper on PFAS concentrations in monitoring data from across Europe.

The code processes geospatial PFAS concentration data, applies the filters and calculations described in the manuscript, and produces summary results in Excel format and Figures. The Excel format output can be modified to any preferred non-proprietary file format.

The repository is intended to support transparency and reproducibility of the research findings. It should be read alongside the associated manuscript.

## How to run the code

### Requirements

The analysis requires:

- Python 3.13 or later
- The input data files described below
- Sufficient disk space for the input data and generated outputs

The analysis has been developed for execution from the root directory of this repository.

### 1. Install Python

Install Python 3.13 or a later version from:

<https://www.python.org/downloads/>

On Windows, select the following option during installation:

```bash
Add Python to PATH
```

Confirm that Python has been installed:

```cmd
python --version
```

On some systems, the command may be:

```bash
python3 --version
```

### 2. Create a virtual environment

A virtual environment keeps the dependencies for this project separate from other Python installations and projects.

From the root directory of the repository, create a virtual environment:

#### Windows

```bash
python -m venv .venv
```

Activate it with:

```bash
.venv\Scripts\activate
```

#### macOS or Linux

```bash
python3 -m venv .venv
```

Activate it with:

```bash
source .venv/bin/activate
```

When the environment is active, `(.venv)` should appear at the beginning of the command prompt.

### 3. Install the project and its dependencies

With the virtual environment activated, upgrade the Python packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the project and its dependencies from the repository root:

```bash
pip install .
```

The projects' dependencies are defined in `pyproject.toml`.

### 4. Prepare the input data

Place the required input data in the directory expected by the analysis code. The input files are described in the respective section.

Before running the analysis, check that:

- the input Parquet is present
- all required shapefile components are present
- the file and directory names match those used in the code and
- the working directory is the repository root.

### 5. Run the analysis

Run the analysis script using the command-line entry point as defined in `pyproject.toml`:

```bash
run-pfas-analysis \
-I <path_to_input_file_directory> \
-O <path_to_output_directory>
```

Alternatively, `make` can be used by running:

```bash
make analysis
```

Here, `data/input` and `results` are predefined as input and output paths.

The analysis should create an Excel workbook as well as Figures in the designated output directory. The exact filename and location are determined by the output path defined in the analysis script. Additionally, a log file is create in a `logs` directory.

## Input data directory

Following files must be present in the `data/input` directory before execution of the code. Licenses of the respective dataset apply.

| File name                | Description                                                                   | Reference                                           |
| ------------------------ | ----------------------------------------------------------------------------- | --------------------------------------------------- |
| `pfas_data.parquet`      | Parquet-file containing daily PFAS concentration data                         | [PFAS concentration data](#pfas-concentration-data) |
| `hybas_eu_lev04_v1c.shp` | ShapeFile containing catchment delineations (level 04) (Lehner & Grill, 2013) | [References](#references)                           |
| `hybas_eu_lev12_v1c.shp` | ShapeFile containing catchment delineations (level 12) (Lehner & Grill, 2013) | [References](#references)                           |

### PFAS concentration data

The file `pfas_data.parquet` contains daily PFAS concentration data compiled from monitoring sites across Europe. The dataset is compiled from existing compilations, research data, and data from authoritive sources. Since dynamic datasets were used, which are regularly updated, we provide a snapshot of the used dataset in `data/concentrations_datasets.zip`. The compilation process is documented in the executable notebook `compilation_of_dataset.ipynb` in the `scripts` folder. Licensing information and doi of the respective datasets are provided in `licence-mapping.toml`. For the creation of `pfas_data.parquet` following files are also needed ot be placed in `data/auxiliary`:

| File name                      | Description                                                                   | Reference                                                             |
| ------------------------------ | ----------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `ne_10m_admin_0_countries.shp` | Country shape file (v5.1.1) by Natural Earth (n.d.)                           | [Website](https://www.naturalearthdata.com) [References](#references) |
| `hybas_eu_lev01_v1c.shp`       | ShapeFile containing catchment delineations (level 01) (Lehner & Grill, 2013) | [References](#references)                                             |


The compiled dataset, i.e. `pfas_data.parquet` is readily provided in `data/input`. The compilation is described in the original research article.


The file contains information including:

| Field          | Type         | Description                                                                                                    |
| -------------- | ------------ | -------------------------------------------------------------------------------------------------------------- |
| `date`         | `datetime64` | Date of sampling                                                                                               |
| `substance`    | `string`     | PFAS substance name                                                                                            |
| `conc`         | `float64`    | Measured concentration in ng/L                                                                                 |
| `unit`         | `str`        | Unit of the concentration (harmonised to ng/L)                                                                 |
| `less_than`    | `boolean`    | Indicator for observations below the reporting or detection limit. `True` equals observations below the limit. |
| `geometry`     | `geometry`   | Spatial location of the observation (EPSG:4326)                                                                |
| `dataset_name` | `str`        | Hints at the original dataset. See `licence-mapping.toml` for details.                                         |
| `HYBAS_ID04`   | `float64`    | HydroBASIN ID of level 04 basins (see Shapefiles section)                                                      |
| `basin`        | `str`        | Basin name derived from major rivers in a given level 04 basin                                                 |
| `lat`          | `float64`    | Latitude                                                                                                       |
| `lon`          | `float64`    | Longitude                                                                                                      |

Most fields are adopted from the PFAS datahub (Cordner, 2024). The PFAS datahub provides a `country` field, which we decided to re-compute during the analysis to apply one consistent method.

The data represent the harmonised dataset used for the analyses presented in the associated manuscript. Daily records should be interpreted as observations reported or aggregated at the daily level. They do not necessarily represent continuous measurements throughout each day. The data may contain observations below the analytical limit of detection. These records should not automatically be interpreted as measurements of zero concentration. The treatment of non-detects and values below the limit of detection is described in the manuscript and in the accompanying analysis code.

### Shapefiles

The HydroBASINS datasets require their associated metadata and component files. A shapefile is not a single file: the `.shp` file must normally be accompanied by files such as:

```plain
.shx
.dbf
.prj
```

For example, the HydroBASINS level 12 dataset may include:

```plain
hybas_eu_lev12_v1c.shp
hybas_eu_lev12_v1c.shx
hybas_eu_lev12_v1c.dbf
hybas_eu_lev12_v1c.prj
```

The same applies to any other shapesfiles.

Please refer to the original data source and the References section for the relevant metadata and licensing information.

## References

Cordner, A., Brown, P., Cousins, I. T., Scheringer, M., Martinon, L., Dagorn, G., Aubert, R., Hosea, L., Salvidge, R., Felke, C., Tausche, N., Drepper, D., Liva, G., Tudela, A., Delgado, A., Salvatore, D., Pilz, S., & Horel, S. (2024). PFAS Contamination in Europe: Generating Knowledge and Mapping Known and Likely Contamination with “Expert-Reviewed” Journalism. *Environmental Science & Technology*, *58*(15), 6616–6627. [https://doi.org/10.1021/acs.est.3c09746](https://doi.org/10.1021/acs.est.3c09746)

Lehner, B., & Grill, G. (2013). Global river hydrography and network routing: Baseline data and new approaches to study the world’s large river systems. Hydrological Processes, 27(15), 2171–2186. <https://doi.org/10.1002/hyp.9740>

Natural Earth. (n.d.). *Natural Earth vector data, 1:10m scale* (Version 5.1.1) [Dataset]. Retrieved April 23, 2026, from [www.naturalearthdata.com](https://doi.org/www.naturalearthdata.com)
