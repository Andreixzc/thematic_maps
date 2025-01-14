import ee
import geemap
import solara
import ipywidgets as widgets
import sqlite3
ee.Initialize()

class Map(geemap.Map):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_basemap("satellite")

        # Create text input for reservoir name
        self.reservoir_name_input = widgets.Text(
            placeholder="Enter reservoir name",
            description="Name:"
        )

        # Dropdown to display the registered reservoirs
        self.reservoir_dropdown = widgets.Dropdown(
            options=self.get_registered_reservoirs(),
            description='Reservatórios:',
            disabled=False,
        )
        # Add event handler to handle selection change
        self.reservoir_dropdown.observe(self.on_reservoir_selected, names='value')

        # Create save button
        clip_btn = widgets.Button(description="Salvar reservatório")
        clip_btn.on_click(self.on_clip_btn_clicked)

        # Add the text input, button, and dropdown to the widget
        widget = widgets.VBox([self.reservoir_name_input, clip_btn, self.reservoir_dropdown])
        self.add_widget(widget, position="bottomright")
        self.last_roi_cords = None

    def on_clip_btn_clicked(self, b):
        if self.user_roi is not None:
            try:
                roi_info = self.user_roi.getInfo()
                coords = roi_info['coordinates']
                self.last_roi_cords = coords

                # Get the reservoir name from the text input
                reservoir_name = self.reservoir_name_input.value
                if not reservoir_name:
                    print("Reservoir name is required!")
                    return

                # Save the coordinates and reservoir name to the database
                self.save_coords_db(reservoir_name, coords)
                
                # Update the dropdown with the new reservoir
                self.reservoir_dropdown.options = self.get_registered_reservoirs()

                print("Last ROI coordinates:", self.last_roi_cords)
            except Exception as e:
                print(f"Error clipping image: {e}")

    def on_reservoir_selected(self, change):
        """Handles reservoir selection from the dropdown and displays the area on the map."""
        selected_reservoir = change['new']
        if selected_reservoir:
            # Fetch the coordinates from the database for the selected reservoir
            coords = self.get_coords_for_reservoir(selected_reservoir)
            if coords:
                # Create a new layer for the selected reservoir area
                roi = ee.Geometry.Polygon(coords)
                self.addLayer(roi, {"color": "blue", "opacity": 0.5}, f"Reservoir: {selected_reservoir}")
                self.centerObject(roi)  # Center map on the selected ROI

    def get_coords_for_reservoir(self, reservoir_name):
        """Fetch the coordinates for a specific reservoir from the database."""
        try:
            conn = sqlite3.connect('Banco/water-quality.db')
            cursor = conn.cursor()
            cursor.execute('SELECT coordenadas FROM reservatorios WHERE nome = ?', (reservoir_name,))
            result = cursor.fetchone()
            conn.close()
            if result:
                # Convert the coordinates back from string to list
                return eval(result[0])
            else:
                print(f"No coordinates found for reservoir: {reservoir_name}")
                return None
        except Exception as e:
            print(f"Error fetching coordinates for reservoir {reservoir_name}: {e}")
            return None

    def get_registered_reservoirs(self):
        """Fetch the list of registered reservoirs from the database."""
        try:
            conn = sqlite3.connect('Banco/water-quality.db')
            cursor = conn.cursor()
            cursor.execute('SELECT nome FROM reservatorios')
            rows = cursor.fetchall()
            conn.close()
            # Extract reservoir names from the query result
            return [row[0] for row in rows]
        except Exception as e:
            print(f"Error fetching reservoir names: {e}")
            return []

    def save_coords_db(self, nome, coords):
        try:
            conn = sqlite3.connect('Banco/water-quality.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reservatorios (nome, coordenadas)
                VALUES (?, ?)
            ''', (nome, str(coords)))
            conn.commit()
            conn.close()
            print("Coordinates saved successfully!")
        except Exception as e:
            print(f"Error saving coordinates: {e}")

@solara.component
def Page():
    with solara.Column(style={"min-width": "500px"}):
        Map.element(
            center=[40, -100],
            zoom=4,
            height="800px",
        )

Page()
