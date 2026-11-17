# Data sources and access

This repository contains **no raw or site-level data**. Obtain the inputs below from their providers and place them where the pipeline expects them. The paths are relative to `pipeline/`.

| Dataset | Provider and licence | Used for | Expected location |
|---|---|---|---|
| Water Quality Archive (now the Water Quality Explorer), reactive orthophosphate as P, determinand 0180, 2023–2024 | Environment Agency, Open Government Licence v3.0 | Measurements | Downloaded by `pilot_01/scripts/01_download_wqa.py` from `https://environment.data.gov.uk/water-quality` (observation API) into `pilot_01/raw/` |
| River Surveillance Network Analysis Ready Data (NCEA), `NCEA_River_Surveillance_Network_ARD_Open.zip` | Environment Agency, OGL v3.0 | Official RSN site list | `pilot_01/raw/rsn_ard/`; download from the Environment Agency data portal ([file](https://environment.data.gov.uk/api/file/download?fileDataSetId=68444e44-461e-45f8-a7f8-d5296eaf2da9&fileName=NCEA_River_Surveillance_Network_ARD_Open.zip)) |
| Event Duration Monitoring: Storm Overflows: Annual Returns (2024 return), `EDM_2024_Storm_Overflow_Annual_Return.zip` | Environment Agency, OGL v3.0 | Storm-overflow covariate | `pilot_01/raw/edm/`; available from data.gov.uk |
| WFD River Water Body Catchments, Cycle 3 (Classification 2022), full-resolution GeoPackage | Environment Agency, OGL v3.0 | Water bodies and geology typology | `phase_02/data_raw/wfd/extracted/`; download from the Environment Agency data portal ([file](https://environment.data.gov.uk/api/file/download?fileDataSetId=22f7760d-5e6c-469e-92c2-b2ae729c4ff8&fileName=Water_Framework_Directive_WFD_River_Water_Body_Catchments_Cycle_3_Classification_2022_Full_Resolution.gpkg.zip)) |
| Administrative Boundaries: Water Management Areas | Environment Agency, OGL v3.0 | Management hierarchy | `phase_02/data_raw/ea_areas/extracted/` ([file](https://environment.data.gov.uk/api/file/download?fileDataSetId=c348109d-324e-4e50-a8e4-3b4003bde0d9&fileName=Administrative_Boundaries_Water_Management_Areas.gpkg.zip)) |
| OS Open Rivers (GeoPackage, `oprvrs_gb.gpkg`) | Ordnance Survey OpenData, OGL v3.0. Contains OS data © Crown copyright and database right. | River network | `phase_02/data_raw/os_open_rivers/extracted/Data/`; download from the OS Data Hub (OpenData) |
| **NRFA station information** (gauged catchment areas) | UK Centre for Ecology & Hydrology, **NRFA click-through licence. Not redistributed.** | River-size calibration and flow-direction check | `phase_02/data_raw/nrfa/nrfa_station_info.json`. Retrieve it yourself from the NRFA web service after accepting its terms. The query used was `https://nrfaapps.ceh.ac.uk/nrfa/ws/station-info?station=*&format=json-object&fields=id,name,river,location,easting,northing,catchment-area,grid-reference,lat-long,station-level,measuring-authority-id,hydrometric-area,opened,closed,bfihost,feh-pooling,factors-affecting-runoff` (retrieved 2026-11-10). Acknowledgement: Data from the UK National River Flow Archive. |

The measurements were retrieved on 10 November 2026. Providers may revise their archives, so a new download can differ slightly from the data analysed.

## Not included, and why

- **NRFA station records:** the licence does not permit redistribution.
- **Raw and processed site-level and site-year tables, the river frame, the Stan `data.json`, posterior draws and per-draw scores.** They are excluded for two reasons:
  - the river frame and the size covariate are derived using NRFA records, and whether those derived values may be redistributed has not been confirmed;
  - the files are large.
- **Planned deposit:** the authors plan to deposit derived analysis tables separately, with a persistent identifier, as stated in the paper's Data availability section.
