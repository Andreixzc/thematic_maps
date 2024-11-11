import ee
import geemap
import joblib
import numpy as np
import os
from datetime import datetime
import threading
from threading import Lock
import rasterio
from rasterio.merge import merge

class ChlorophyllProcessor:
    def __init__(self, model_path, scaler_path):
        """Initialize the processor with model and scaler paths"""
        # Initialize Earth Engine
        try:
            ee.Initialize()
        except Exception as e:
            ee.Authenticate()
            ee.Initialize()
            
        # Load model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Define feature names
        self.feature_names = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'NDCI', 'NDVI', 'FAI', 
                            'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio', 'Month', 'Season']

    def process_tile(self, tile_geometry, date_range):
        """Process a single tile of the area of interest"""
        # Sentinel-2 collection for tile
        sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(tile_geometry) \
            .filterDate(date_range[0], date_range[1]) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # Get median image
        image = sentinel2.median().clip(tile_geometry)
        
        # Select bands of interest
        bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
        image = image.select(bands)
        
        # Create water mask
        MNDWI = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        water_mask = MNDWI.gt(0.3)
        
        # Calculate indices
        NDCI = image.normalizedDifference(['B5', 'B4']).rename('NDCI')
        NDVI = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        FAI = image.expression(
            'NIR - (RED + (SWIR - RED) * (NIR_wl - RED_wl) / (SWIR_wl - RED_wl))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'SWIR': image.select('B11'),
                'NIR_wl': 842,
                'RED_wl': 665,
                'SWIR_wl': 1610
            }
        ).rename('FAI')
        
        # Calculate band ratios
        B3_B2_ratio = image.select('B3').divide(image.select('B2')).rename('B3_B2_ratio')
        B4_B3_ratio = image.select('B4').divide(image.select('B3')).rename('B4_B3_ratio')
        B5_B4_ratio = image.select('B5').divide(image.select('B4')).rename('B5_B4_ratio')
        
        # Get date information
        middle_date = ee.Date(sentinel2.limit(1).first().get('system:time_start'))
        month = ee.Image.constant(middle_date.get('month')).rename('Month')
        season = ee.Image.constant(middle_date.get('month').add(2).divide(3).floor().add(1)).rename('Season')
        
        # Combine all features
        image_with_indices = image.addBands([NDCI, NDVI, FAI, B3_B2_ratio, B4_B3_ratio, 
                                           B5_B4_ratio, month, season])
        
        # Create scaled bands
        scaled_bands = []
        for i, name in enumerate(self.feature_names):
            scaled_band = image_with_indices.select(name) \
                .subtract(ee.Number(self.scaler.mean_[i])) \
                .divide(ee.Number(self.scaler.scale_[i])) \
                .rename(f'scaled_{name}')
            scaled_bands.append(scaled_band)
        
        # Combine scaled bands
        scaled_image = ee.Image.cat(scaled_bands)
        
        # Create prediction
        weighted_bands = []
        for i, (name, importance) in enumerate(zip(self.feature_names, self.model.feature_importances_)):
            weighted_band = scaled_image.select(f'scaled_{name}').multiply(ee.Number(importance))
            weighted_bands.append(weighted_band)
        
        predicted_image = ee.Image.cat(weighted_bands).reduce(ee.Reducer.sum()).rename('chlorophyll_pred')
        
        # Set non-water areas to -9999
        final_image = predicted_image.where(water_mask.Not(), ee.Image.constant(-9999))
        
        return final_image

    def merge_tiff_files(self, directory, pattern, output_file):
        """Merge multiple TIFF files into a single file"""
        tiff_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                      if f.startswith(pattern) and f.endswith('.tif')]
        
        if not tiff_files:
            raise ValueError("No TIFF files found to merge")
        
        src_files_to_mosaic = []
        try:
            for tiff in tiff_files:
                src = rasterio.open(tiff)
                src_files_to_mosaic.append(src)
            
            mosaic, out_trans = merge(src_files_to_mosaic)
            
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "nodata": -9999  # Explicitly set the no-data value
            })
            
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            return tiff_files
        finally:
            for src in src_files_to_mosaic:
                src.close()

    def process_reservoir(self, reservoir_name, aoi, date_range, n_tiles=4, output_dir='outputs'):
        """Process a reservoir and save results"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for tiles
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get AOI bounds
        aoi_bounds = aoi.bounds().coordinates().getInfo()[0]
        xmin, ymin = aoi_bounds[0][0], aoi_bounds[0][1]
        xmax, ymax = aoi_bounds[2][0], aoi_bounds[2][1]
        
        # Calculate tile sizes
        x_step = (xmax - xmin) / n_tiles
        y_step = (ymax - ymin) / n_tiles
        
        lock = Lock()
        tile_results = []
        processed_files = []
        
        def process_and_save_tile(i, j):
            # Create tile geometry
            x0 = xmin + i * x_step
            x1 = xmin + (i + 1) * x_step
            y0 = ymin + j * y_step
            y1 = ymin + (j + 1) * y_step
            tile_geometry = ee.Geometry.Polygon([[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]])
            
            try:
                # Process tile
                tile_result = self.process_tile(tile_geometry, date_range)
                
                # Save tile result
                out_file = os.path.join(temp_dir, f'Tile_{i+1}_{j+1}.tif')
                with lock:
                    geemap.ee_export_image(
                        tile_result,
                        filename=out_file,
                        scale=30,
                        region=tile_geometry
                    )
                    print(f'Tile {i+1}_{j+1} processed and saved: {out_file}')
                    tile_results.append(tile_result)
                    processed_files.append(out_file)
            except Exception as e:
                print(f"Error processing tile {i+1}_{j+1}: {str(e)}")
        
        # Process tiles in parallel
        threads = []
        for i in range(n_tiles):
            for j in range(n_tiles):
                t = threading.Thread(target=process_and_save_tile, args=(i, j))
                threads.append(t)
                t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        if not tile_results:
            raise RuntimeError("No tiles were successfully processed")
        
        # Generate output filename with date
        date_str = datetime.strptime(date_range[0], '%Y-%m-%d').strftime('%Y%m%d')
        output_filename = f"{reservoir_name}_{date_str}_chlorophyll.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Merge tiles
        print("Merging tiles...")
        tile_files = self.merge_tiff_files(temp_dir, 'Tile', output_path)
        
        # Calculate and save min/max values, excluding -9999 values
        merged_result = ee.ImageCollection(tile_results).mosaic()
        
        # Create mask for valid values (not -9999)
        valid_mask = merged_result.neq(-9999)
        masked_result = merged_result.updateMask(valid_mask)
        
        min_max_values = masked_result.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Save min/max values to text file
        stats_filename = f"{reservoir_name}_{date_str}_stats.txt"
        stats_path = os.path.join(output_dir, stats_filename)
        with open(stats_path, 'w') as f:
            f.write(f"Reservoir: {reservoir_name}\n")
            f.write(f"Date Range: {date_range[0]} to {date_range[1]}\n")
            f.write(f"Minimum chlorophyll value: {min_max_values['chlorophyll_pred_min']}\n")
            f.write(f"Maximum chlorophyll value: {min_max_values['chlorophyll_pred_max']}\n")
        
        # Clean up temporary files
        for file in tile_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {str(e)}")
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error removing temporary directory: {str(e)}")
        
        return output_path, stats_path

# Example usage:
if __name__ == "__main__":
    # Initialize processor
    processor = ChlorophyllProcessor(
        model_path='models/cacu/clorofila/model.joblib',
        scaler_path='models/cacu/clorofila/scaler.joblib'
    )
    
    # Define area of interest (example coordinates)
    aoi = ee.Geometry.Polygon([[[-45.559114, -18.954365], [-45.559114, -18.212409], 
                               [-44.839706, -18.212409], [-44.839706, -18.954365], 
                               [-45.559114, -18.954365]]])
    
    # Process reservoir
    output_tiff, output_stats = processor.process_reservoir(
        reservoir_name="Tres_Marias",
        aoi=aoi,
        date_range=['2020-01-01', '2020-04-01'],
        output_dir='chlorophyll_analysis'
    )
    
    print(f"Processing complete!")
    print(f"GeoTIFF saved to: {output_tiff}")
    print(f"Statistics saved to: {output_stats}")