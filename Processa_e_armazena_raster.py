import ee
import geemap
import joblib
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import threading
from threading import Lock

# Inicializar Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

class WaterQualityPredictor:
    def __init__(self):
        """Initialize Earth Engine and define basic parameters"""
        self.bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
        self.feature_columns = [
            'B2', 'B3', 'B4', 'B5', 'B8', 'B11',
            'NDCI', 'NDVI', 'FAI', 'MNDWI',
            'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio',
            'Month', 'Season'
        ]

    def process_satellite_image(self, aoi, start_date, end_date, cloud_threshold=20):
        """Process Sentinel-2 imagery for the area of interest"""
        # Get Sentinel-2 collection
        sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)) \
            .map(lambda image: image.clip(aoi))

        if sentinel2.size().getInfo() == 0:
            raise ValueError(f"No images found for the specified period and cloud coverage threshold")

        # Get median image
        image = sentinel2.select(self.bands).median()
        
        # Get middle date
        middle_date = sentinel2.sort('system:time_start').toList(sentinel2.size()).get(sentinel2.size().divide(2).floor())
        middle_date = ee.Image(middle_date).date()
        image = image.set('system:time_start', middle_date.millis())

        # Calculate water mask
        MNDWI = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        water_mask = MNDWI.gt(0.3)
        image = image.updateMask(water_mask)

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

        # Calculate ratios
        B3_B2_ratio = image.select('B3').divide(image.select('B2')).rename('B3_B2_ratio')
        B4_B3_ratio = image.select('B4').divide(image.select('B3')).rename('B4_B3_ratio')
        B5_B4_ratio = image.select('B5').divide(image.select('B4')).rename('B5_B4_ratio')

        # Add temporal features
        date = ee.Date(image.get('system:time_start'))
        month = ee.Image.constant(date.get('month')).rename('Month')
        season = ee.Image.constant(date.get('month').add(2).divide(3).floor().add(1)).rename('Season')

        # Combine all features
        return image.addBands([NDCI, NDVI, FAI, MNDWI, B3_B2_ratio, B4_B3_ratio, B5_B4_ratio, month, season])

    def predict_parameter(self, image_with_features, model, scaler, water_mask):
        """Apply the model to predict water quality parameter"""
        features = image_with_features.select(self.feature_columns)
        
        # Converter arrays NumPy para listas Python
        mean_values = scaler.mean_.tolist()
        scale_values = scaler.scale_.tolist()
        
        # Scale features
        scaled_features = features.subtract(ee.Image.constant(mean_values)).divide(ee.Image.constant(scale_values))
        
        # Handle different types of models
        if hasattr(model, 'coef_'):  # Linear models
            coefficients = model.coef_.tolist() if isinstance(model.coef_, np.ndarray) else model.coef_
            intercept = float(model.intercept_) if isinstance(model.intercept_, np.ndarray) else model.intercept_
            
            expressions = [scaled_features.select(name).multiply(float(coef)) 
                        for name, coef in zip(self.feature_columns, coefficients)]
            predicted = ee.Image.constant(intercept).add(
                ee.Image.cat(expressions).reduce(ee.Reducer.sum())
            )
            
        elif hasattr(model, 'feature_importances_'):  # Tree-based models
            coefficients = model.feature_importances_.tolist() if isinstance(model.feature_importances_, np.ndarray) else model.feature_importances_
            
            expressions = [scaled_features.select(name).multiply(float(coef)) 
                        for name, coef in zip(self.feature_columns, coefficients)]
            predicted = ee.Image.cat(expressions).reduce(ee.Reducer.sum())
            
        else:
            raise NotImplementedError(f"Model type {type(model).__name__} not yet supported")

        return predicted.updateMask(water_mask)

    def export_prediction(self, predicted_image, aoi, output_dir, parameter_name, n_tiles=2, scale=30):
        """Export the prediction as GeoTIFF files"""
        os.makedirs(output_dir, exist_ok=True)
        
        tile_list = []
        lock = Lock()
        
        self._split_aoi_and_export(
            aoi=aoi,
            n_tiles=n_tiles,
            scale=scale,
            image=predicted_image,
            output_dir=output_dir,
            parameter_name=parameter_name,
            lock=lock,
            tile_list=tile_list
        )
        
        return tile_list

    def _split_aoi_and_export(self, aoi, n_tiles, scale, image, output_dir, parameter_name, lock, tile_list):
        """Split area of interest into tiles and export them in parallel"""
        aoi_bounds = aoi.bounds().coordinates().getInfo()[0]
        xmin, ymin = aoi_bounds[0][0], aoi_bounds[0][1]
        xmax, ymax = aoi_bounds[2][0], aoi_bounds[2][1]
        x_step = (xmax - xmin) / n_tiles
        y_step = (ymax - ymin) / n_tiles

        def export_tile(i, j):
            x0 = xmin + i * x_step
            x1 = xmin + (i + 1) * x_step
            y0 = ymin + j * y_step
            y1 = ymin + (j + 1) * y_step
            
            tile = ee.Geometry.Polygon([[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]])
            tile_image = image.clip(tile)
            tile_list.append(tile_image)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_file = os.path.join(
                output_dir, 
                f'{parameter_name}_Tile_{i+1}_{j+1}_{timestamp}.tif'
            )
            
            lock.acquire()
            try:
                geemap.ee_export_image(
                    tile_image, 
                    filename=out_file,
                    scale=scale,
                    region=tile
                )
                print(f'Exported: {out_file}')
            except Exception as e:
                print(f"Error exporting tile {i+1}_{j+1}: {str(e)}")
            finally:
                lock.release()

        threads = []
        for i in range(n_tiles):
            for j in range(n_tiles):
                t = threading.Thread(target=export_tile, args=(i, j))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

def process_reservoirs(reservoirs, date_range, output_base_dir='predictions'):
    """Process multiple reservoirs and parameters"""
    predictor = WaterQualityPredictor()
    
    for reservoir_name, geometry in reservoirs.items():
        print(f"Processing reservoir: {reservoir_name}")
        
        reservoir_model_dir = Path(f'models/{reservoir_name}')
        if not reservoir_model_dir.exists():
            print(f"No models found for reservoir: {reservoir_name}")
            continue
            
        reservoir_output_dir = Path(output_base_dir) / reservoir_name
        
        for parameter_dir in reservoir_model_dir.iterdir():
            if parameter_dir.is_dir():
                parameter_name = parameter_dir.name
                print(f"Processing parameter: {parameter_name}")
                
                try:
                    # Load model and scaler
                    model = joblib.load(parameter_dir / 'model.joblib')
                    scaler = joblib.load(parameter_dir / 'scaler.joblib')
                    
                    # Process satellite image
                    image_with_features = predictor.process_satellite_image(
                        geometry, 
                        date_range[0], 
                        date_range[1]
                    )
                    
                    # Get water mask
                    water_mask = image_with_features.select('MNDWI').gt(0.3)
                    
                    # Make prediction
                    predicted_image = predictor.predict_parameter(
                        image_with_features,
                        model,
                        scaler,
                        water_mask
                    )
                    
                    # Export prediction
                    parameter_output_dir = reservoir_output_dir / parameter_name
                    predictor.export_prediction(
                        predicted_image,
                        geometry,
                        parameter_output_dir,
                        parameter_name
                    )
                    
                except Exception as e:
                    print(f"Error processing {parameter_name} for {reservoir_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()

def main():
    # Define reservoirs
    reservoirs = {
        'cacu': ee.Geometry.Polygon([[
            [-51.230662, -18.538214],
            [-51.230662, -18.420665],
            [-51.134215, -18.420665],
            [-51.134215, -18.538214],
            [-51.230662, -18.538214]
        ]]),
        'tres_marias': ee.Geometry.Polygon([[
            [-45.559114, -18.954365],
            [-45.559114, -18.212409],
            [-44.839706, -18.212409],
            [-44.839706, -18.954365],
            [-45.559114, -18.954365]
        ]])
    }
    
    # Define date range
    date_range = ('2020-01-01', '2020-04-01')
    
    # Process all reservoirs
    process_reservoirs(reservoirs, date_range)

if __name__ == "__main__":
    main()