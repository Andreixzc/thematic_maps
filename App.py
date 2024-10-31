import geemap
import rasterio
import glob
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
import geopandas as gpd

class WaterQualityMapVisualizer:
    def __init__(self, predictions_dir='predictions'):
        """
        Initialize the map visualizer
        
        Args:
            predictions_dir (str): Directory containing the prediction TIF files
        """
        self.predictions_dir = Path(predictions_dir)
        self.parameters = {}
        self.current_parameter = None
        self.map = None
        
        # Paletas de cores para diferentes parâmetros
        self.color_palettes = {
            'Clorofila': ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', 
                         '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026'],
            'Turbidez': ['#440154', '#482878', '#3E4989', '#31688E', '#26828E',
                        '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
            'Transparencia': ['#A50026', '#D73027', '#F46D43', '#FDAE61', '#FEE090', 
                            '#E0F3F8', '#ABD9E9', '#74ADD1', '#4575B4', '#313695'],
            'Solidos_dissolvidos_totais': ['#000004', '#170B3A', '#420A68', '#6B176E',
                                          '#932667', '#BB3654', '#DD513A', '#F37819',
                                          '#FCA50A', '#F6D746']
        }
        
        self.load_data()
    
    def load_data(self):
        """Load all TIF files from the predictions directory"""
        for parameter_dir in self.predictions_dir.iterdir():
            if parameter_dir.is_dir():
                tif_files = list(parameter_dir.glob('**/*.tif'))
                if tif_files:
                    self.parameters[parameter_dir.name] = {
                        'files': tif_files,
                        'stats': self._calculate_stats(tif_files)
                    }
    
    def _calculate_stats(self, tif_files):
        """Calculate min and max values across all tiles"""
        all_mins = []
        all_maxs = []
        
        for tif_file in tif_files:
            with rasterio.open(tif_file) as src:
                data = src.read(1)
                data = data[data != src.nodata]
                if len(data) > 0:
                    all_mins.append(np.nanmin(data))
                    all_maxs.append(np.nanmax(data))
        
        return {
            'min': min(all_mins) if all_mins else 0,
            'max': max(all_maxs) if all_maxs else 1
        }
    
    def create_map(self, center=None, zoom=None):
        """Create a new map instance"""
        self.map = geemap.Map(center=center, zoom=zoom)
        self.map.add_basemap('SATELLITE')
        return self.map
    
    def add_parameter_layer(self, parameter_name, opacity=1.0):
        """Add a parameter layer to the map"""
        if parameter_name not in self.parameters:
            print(f"Parameter {parameter_name} not found")
            return
        
        self.current_parameter = parameter_name
        stats = self.parameters[parameter_name]['stats']
        
        # Criar visualização para o parâmetro
        vis_params = {
            'min': stats['min'],
            'max': stats['max'],
            'palette': self.color_palettes.get(parameter_name, ['blue', 'red'])
        }
        
        # Adicionar cada tile ao mapa
        for tif_file in self.parameters[parameter_name]['files']:
            layer_name = f"{parameter_name}_{tif_file.stem}"
            self.map.add_raster(str(tif_file), 
                              layer_name=layer_name,
                              opacity=opacity,
                              **vis_params)
    
    def add_legend(self, title=None):
        """Add a legend to the map"""
        if not self.current_parameter:
            return
        
        stats = self.parameters[self.current_parameter]['stats']
        palette = self.color_palettes.get(self.current_parameter, ['blue', 'red'])
        
        legend_title = title or self.current_parameter
        legend_dict = {
            'min': stats['min'],
            'max': stats['max'],
            'palette': palette,
            'add_colorbar': True,
            'label': legend_title
        }
        
        self.map.add_legend(title=legend_title, legend_dict=legend_dict)
    
    def add_shapefile(self, shapefile_path, style=None):
        """Add a shapefile to the map"""
        if style is None:
            style = {
                'color': 'white',
                'fillColor': 'transparent',
                'width': 2
            }
        
        self.map.add_shapefile(shapefile_path, style=style, layer_name='Boundary')

def create_interactive_map(predictions_dir='predictions', shapefile_path=None):
    """
    Create an interactive map with widgets for parameter selection and opacity control
    
    Args:
        predictions_dir (str): Directory containing the prediction TIF files
        shapefile_path (str): Optional path to a shapefile for reservoir boundary
    """
    visualizer = WaterQualityMapVisualizer(predictions_dir)
    
    # Criar o mapa
    m = visualizer.create_map()
    
    # Adicionar shapefile se fornecido
    if shapefile_path and os.path.exists(shapefile_path):
        visualizer.add_shapefile(shapefile_path)
    
    # Criar widgets
    parameter_dropdown = widgets.Dropdown(
        options=list(visualizer.parameters.keys()),
        description='Parameter:',
        style={'description_width': 'initial'}
    )
    
    opacity_slider = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=1.0,
        step=0.1,
        description='Opacity:',
        style={'description_width': 'initial'}
    )
    
    def update_map(parameter, opacity):
        """Atualizar o mapa com base nos controles"""
        # Limpar camadas existentes
        m.clear_layers()
        m.add_basemap('SATELLITE')
        
        # Adicionar nova camada
        visualizer.add_parameter_layer(parameter, opacity)
        
        # Adicionar shapefile novamente se existir
        if shapefile_path and os.path.exists(shapefile_path):
            visualizer.add_shapefile(shapefile_path)
        
        # Atualizar legenda
        visualizer.add_legend(f'{parameter} Distribution')
    
    # Criar interface interativa
    interact(update_map, 
            parameter=parameter_dropdown,
            opacity=opacity_slider)
    
    return m

# Exemplo de uso em notebook
if __name__ == "__main__":
    # Criar mapa interativo
    predictions_dir = 'predictions'  # ajuste conforme necessário
    shapefile_path = 'path/to/reservoir_boundary.shp'  # opcional
    
    map_display = create_interactive_map(predictions_dir, shapefile_path)
    display(map_display)