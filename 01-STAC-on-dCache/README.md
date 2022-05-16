# Work with STAC Catalogs on the dCache Storage

This example includes two Jupyter notebooks that illustrate how to:
* search for scenes from [the Sentinel-2 mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) as part of [a open dataset on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/)
* save the metadata in the form of a [SpatioTemporal Asset Catalog](https://stacspec.org) on the [SURF dCache storage system](http://doc.grid.surfsara.nl/en/latest/Pages/Advanced/grid_storage.html).
* retrieve some of the scenes' assets.
* doing some simple processing on the retrieved assets using a Dask cluster to distribute workload.

## Additional dependencies 

The example requires the packages listed in the [conda environment file](./environment.yaml) provided in this folder.  

