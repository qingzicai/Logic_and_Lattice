---
title: Utilizing the Global ML Building Footprints Dataset to Generate a GeoJSON File for Your GIS Project
date: "2023-10-24T18:30:35Z"
description: "A step-by-step guide"
---

[Global ML Building Footprints Dataset](https://github.com/microsoft/GlobalMLBuildingFootprints)

## Key steps

- Install Python and requisite libraries.
- Use geojson.io to obtain the coordinates of your desired area.
- Utilize Jupyter Notebook to execute the code and generate a GeoJSON file containing the necessary data slices.
- Import the generated GeoJSON file into your project.

## Detailed Procedures

**1. Install Python and requisite libraries:**

1). Install Python via [Anaconda](https://www.anaconda.com/download), ensuring to add it to your system path during installation.</br>
2). Launch Powershell (for Windows) or Terminal (for Mac), and type `conda install geopandas`.</br>
If this command fails due to environment conflicts, attempt `pip install geopandas`. Setting up a new environment may resolve the issue, although it was not successful in my case. Here is how to establish a new environment:

--Create an environment:

```sh
python -m venv myenv
```

--Activate the environment on Windows

```sh
myenv\Scripts\activate
```

--Or Activate the environment on Unix/MacOS:

```sh
source myenv/bin/activate
```

--Install geopandas:

```sh
pip install geopandas
```

3). Install mercantile library

```sh
pip install mercantile
```

**2. Acquiring Coordinates of Target Area Using [geojson.io](http://geojson.io/#map=2/0/20)**</br>
(1). Drawing a Closed Area on the Map:
Utilize the line or polygon tool available on the website to delineate a closed area on the map.</br>
(2). Copying Coordinates:</br>
Copy the coordinates displayed on the right column, as illustrated below.</br>

```sh
    "coordinates": [
        [
        [
            19.69880052797913,
            0.28411636219605896
        ],
        [
            19.69880052797913,
            0.27182233204013073
        ],
        [
            19.713553537814192,
            0.27182233204013073
        ],
        [
            19.713553537814192,
            0.28411636219605896
        ],
        [
            19.69880052797913,
            0.28411636219605896
        ]
        ]
    ],
    "type": "Polygon"
```

**3. Generating a GeoJSON File Using Jupyter Notebook**</br>
(1). Utilize Jupyter Notebook to execute the code and generate a GeoJSON file containing the required data slices </br>
(2). Within Anaconda, launch Jupyter Notebook and create a new Python project.</br> Copy the provided snippet of code into cells within the Notebook, replacing the placeholder coordinates with your own:</br>

```sh
import pandas as pd
import geopandas as gpd
import shapely.geometry
import mercantile
from tqdm import tqdm
import os
import tempfile
import Fiona

# Geometry copied from https://geojson.io
aoi_geom = {
    "coordinates": [
        [
            [-122.16484503187519, 47.69090474454916],
            [-122.16484503187519, 47.6217555345674],
            [-122.06529607517405, 47.6217555345674],
            [-122.06529607517405, 47.69090474454916],
            [-122.16484503187519, 47.69090474454916],
        ]
    ],
    "type": "Polygon",
}
aoi_shape = shapely.geometry.shape(aoi_geom)
minx, miny, maxx, maxy = aoi_shape.bounds

output_fn = "example_building_footprints.geojson"

quad_keys = set()
for tile in list(mercantile.tiles(minx, miny, maxx, maxy, zooms=9)):
    quad_keys.add(int(mercantile.quadkey(tile)))
quad_keys = list(quad_keys)
print(f"The input area spans {len(quad_keys)} tiles: {quad_keys}")

df = pd.read_csv(
    "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
)

idx = 0
combined_rows = []

with tempfile.TemporaryDirectory() as tmpdir:
    # Download the GeoJSON files for each tile that intersects the input geometry
    tmp_fns = []
    for quad_key in tqdm(quad_keys):
        rows = df[df["QuadKey"] == quad_key]
        if rows.shape[0] == 1:
            url = rows.iloc[0]["Url"]

            df2 = pd.read_json(url, lines=True)
            df2["geometry"] = df2["geometry"].apply(shapely.geometry.shape)

            gdf = gpd.GeoDataFrame(df2, crs=4326)
            fn = os.path.join(tmpdir, f"{quad_key}.geojson")
            tmp_fns.append(fn)
            if not os.path.exists(fn):
                gdf.to_file(fn, driver="GeoJSON")
        elif rows.shape[0] > 1:
            raise ValueError(f"Multiple rows found for QuadKey: {quad_key}")
        else:
            raise ValueError(f"QuadKey not found in dataset: {quad_key}")

    # Merge the GeoJSON files into a single file
    for fn in tmp_fns:
        with fiona.open(fn, "r") as f:
            for row in tqdm(f):
                row = dict(row)
                shape = shapely.geometry.shape(row["geometry"])

                if aoi_shape.contains(shape):
                    if "id" in row:
                        del row["id"]
                    row["properties"] = {"id": idx}
                    idx += 1
                    combined_rows.append(row)

schema = {"geometry": "Polygon", "properties": {"id": "int"}}

with fiona.open(output_fn, "w", driver="GeoJSON", crs="EPSG:4326", schema=schema) as f:
    f.writerecords(combined_rows)
```

(3). Executing the Code and Troubleshooting Potential Errors </br>
Execute the code by running the cells. If executed successfully without errors, a new file named example.geojson will be generated in your Jupyter Notebook's directory.</br></br>

##### Possible Error Scenarios:</br>

###### a. Duplicate QuadKey Values in the CSV File:</br>

In this case you may see error message like </br>
`Error Message: ValueError: Multiple rows found for QuadKey: 31312001` </br></br>
--Solution:

Create a new cell and input `print(df[df["QuadKey"] == 31312001])` to identify the duplicate entries. The output should resemble the following:

```{
Location   QuadKey      Url  \
9607    Europe  31312001  https://minedbuildings.blob.core.windows.net/g...
12880  Ireland  31312001  https://minedbuildings.blob.core.windows.net/g...
}
```

If you need the Ireland data, change your code to
`rows = df[(df["QuadKey"] == quad_key) & (df["Location"] == "Ireland")]`

###### b. Missing QuadKey and Importing GeoJSON File.</br>

In this case you may see error message like ` QuadKey not found in dataset: {quad_key}` and the program will stop execution.</br>

--Solution: edit the original code as below

```{
elif rows.shape[0] > 1:
raise ValueError(f"Multiple rows found for QuadKey: {quad_key}")
else:
    print(f"Warning: QuadKey not found in dataset: {quad_key}. Skipping this tile.")
    continue
}
```

This method ensures that upon encountering a QuadKey that is not present in the dataset, a warning message is displayed, allowing the code to proceed to the next QuadKey instead of halting execution and triggering an error.

**4. Import the GeoJSON file in your project.** </br>
Depending on your project's platform, a plugin might be necessary to import the GeoJSON file. To find a suitable plugin, search online using the query app name + GeoJSON file and install the required plugin accordingly.
