# research_pfas_in_europe

## How to run the code


## Input data directory

This directory should contain the following files:

| File name                | Description                                                       | Link                                                 |
| ------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------- |
| `pfas_data.gpkg`         | GeoPackage containing daily PFAS concentration data               | [PFAS concentration data](#pfas-concentration-data)  |
| `hybas_eu_lev12_v1c.shp` | ShapeFile containing catchment delineation (Lehner & Grill, 2013) | [References](#references)                            |
| `hybas_eu_lev04_v1c.shp` | ShapeFile containing catchment delineation (Lehner & Grill, 2013) | [References](#references)                            |
|                          |                                                                   |                                                      |

### HYBAS Shape files

Also needs the metadata files. Please refer to [References](#references).

### PFAS concentration data

The file `pfas_data.gpkg` contains daily PFAS concentration data compiled from monitoring sites across Europe. The data are provided in GeoPackage format and can be opened with GIS software such as QGIS or accessed programmatically using Python libraries including `geopandas` and `fiona`.

The GeoPackage contains the following information:

- sampling-site identifiers;
- sampling dates;
- geographic coordinates;
- PFAS substance names;
- measured concentrations;
- concentration units;
- limits of detection, where available;
- information on censored or non-detect observations; and
- relevant metadata describing the source of each observation.

The data represent the harmonised dataset used for the analyses presented in the associated manuscript. Daily records should be interpreted as observations reported or aggregated at the daily level; they do not necessarily represent continuous measurements throughout each day.

The data may contain observations below the analytical limit of detection. These records should not automatically be interpreted as measurements of zero concentration. The treatment of non-detects and values below the limit of detection is described in the manuscript and in the accompanying analysis code.

The GeoPackage may contain one or more layers. To list the available layers in Python:

```python
import fiona

fiona.listlayers("pfas_data.gpkg")
```

## References

Lehner, B., & Grill, G. (2013). Global river hydrography and network routing: Baseline data and new approaches to study the world’s large river systems. Hydrological Processes, 27(15), 2171–2186. <https://doi.org/10.1002/hyp.9740>
