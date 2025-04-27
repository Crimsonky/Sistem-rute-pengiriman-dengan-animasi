import streamlit as st
import pandas as pd
import networkx as nx
import folium
from folium import Marker
from streamlit_folium import st_folium
import heapq
import time
import tracemalloc
import matplotlib.pyplot as plt
from folium.plugins import AntPath
import datetime
import json
from math import radians, sin, cos, sqrt, atan2
import numpy as np

# ===== Load Data =====
nodes_df = pd.read_csv('transportation_nodes.csv')
edges_df = pd.read_csv('transportation_edges_augmented.csv')

G = nx.DiGraph()
id_to_label = {}
label_to_id = {}

for _, row in nodes_df.iterrows():
    G.add_node(row['id'], label=row['name'], **row.to_dict())
    id_to_label[row['id']] = row['name']
    label_to_id[row['name']] = row['id']

def parse_congestion(cong):
    if pd.isna(cong):
        return 1.0
    try:
        return float(cong)
    except (ValueError, TypeError):
        pass

    s = str(cong).strip().lower()
    mapping = {
        'rendah': 1,
        'sedang': 2,
        'tinggi': 3
    }
    return mapping.get(s, 1.0)

def compute_weight(distance_km, avg_speed_kmh, congestion, direction):
    speed = avg_speed_kmh if avg_speed_kmh and avg_speed_kmh > 0 else 1.0
    cong_val = parse_congestion(congestion)
    time_hours = distance_km / speed
    cong_factor = 1 + cong_val
    dir_factor = 1.5 if direction == 1 else 1.0
    return time_hours * cong_factor * dir_factor

for _, row in edges_df.iterrows():
    weight = compute_weight(
        distance_km=row['distance_km'],
        avg_speed_kmh=row['avg_speed_kmh'],
        congestion=row['congestion'],
        direction=row['direction']
    )
    G.add_edge(
        row['from'],
        row['to'],
        weight=weight,
        label=round(weight, 4),
        **row.to_dict()
    )

def dijkstra(graph, start, end):
    distances = {n: float('inf') for n in graph.nodes}
    distances[start] = 0
    predecessors = {}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        if current_node == end:
            path = []
            while current_node in predecessors:
                path.append(current_node)
                current_node = predecessors[current_node]
            path.append(start)
            path.reverse()
            return path, distances[end]

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return None, float('inf')

def a_star(graph, start, end, heuristic):
    open_set = [(0, start)]
    came_from = {}
    g = {n: float('inf') for n in graph.nodes}
    f = {n: float('inf') for n in graph.nodes}
    g[start] = 0
    f[start] = heuristic(start, end)

    while open_set:
        _, node = heapq.heappop(open_set)
        if node == end:
            path = []
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return list(reversed(path)), g[end]

        for neighbor in graph.neighbors(node):
            tentative_g = g[node] + graph[node][neighbor]['weight']
            if tentative_g < g[neighbor]:
                came_from[neighbor] = node
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f[neighbor], neighbor))
    return None, float('inf')

def heuristic_distance(n1, n2):
    try:
        from math import radians
        from sklearn.metrics.pairwise import haversine_distances
        lat1, lon1 = radians(G.nodes[n1]['latitude']), radians(G.nodes[n1]['longitude'])
        lat2, lon2 = radians(G.nodes[n2]['latitude']), radians(G.nodes[n2]['longitude'])
        return haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1] * 6371
    except:
        return 0

def multi_vehicle_routing(graph, vehicle_data, algorithm):
    routes = {}
    vehicles = [v.copy() for v in vehicle_data]
    
    vehicles_with_idx = sorted(enumerate(vehicles), key=lambda x: x[1]['priority'])
    
    weight_distribution = {}
    
    for vehicle_idx, v in vehicles_with_idx:
        weight_distribution[vehicle_idx] = {
            'main_destination': v['end'],
            'weights': {v['end']: v['weight']}  
        }
    
    for vehicle_idx, v in vehicles_with_idx:
        if v['weight'] > v['capacity']:
            excess_weight = v['weight'] - v['capacity']
            
            orig_distribution = weight_distribution[vehicle_idx]['weights'].copy()
            total_orig_weight = sum(orig_distribution.values())
            
            for dest in orig_distribution:
                orig_distribution[dest] = (orig_distribution[dest] / total_orig_weight) * v['capacity']
            
            weight_distribution[vehicle_idx]['weights'] = orig_distribution
            v['weight'] = v['capacity']  
            
            for other_idx, other_v in vehicles_with_idx:
                if other_idx != vehicle_idx and other_v['start'] == v['start']:
                    available_capacity = other_v['capacity'] - other_v['weight']
                    
                    if available_capacity > 0:  
                        transfer_amount = min(excess_weight, available_capacity)
                        excess_weight -= transfer_amount
                        other_v['weight'] += transfer_amount
                        
                        if 'additional_stops' not in other_v:
                            other_v['additional_stops'] = []
                        if v['end'] not in other_v['additional_stops'] and v['end'] != other_v['end']:
                            other_v['additional_stops'].append(v['end'])
                        
                        if other_idx not in weight_distribution:
                            weight_distribution[other_idx] = {
                                'main_destination': other_v['end'],
                                'weights': {other_v['end']: other_v['weight']}
                            }
                        
                        if v['end'] in weight_distribution[other_idx]['weights']:
                            weight_distribution[other_idx]['weights'][v['end']] += transfer_amount
                        else:
                            weight_distribution[other_idx]['weights'][v['end']] = transfer_amount
                        
                        if excess_weight <= 0:
                            break
    
    for idx, (vehicle_idx, v) in enumerate(vehicles_with_idx):
        if v['weight'] <= v['capacity']:  
            if 'additional_stops' in v and v['additional_stops']:
                full_path = []  
                full_cost = 0
                full_distance = 0
                current_point = v['start']
                
                for stop in v['additional_stops']:
                    path, cost = (dijkstra(graph, current_point, stop) if algorithm == 'dijkstra'
                                else a_star(graph, current_point, stop, heuristic_distance))
                    
                    if path:
                        if full_path:
                            full_path.extend(path[1:]) 
                        else:
                            full_path.extend(path)
                            
                        full_cost += cost
                        segment_distance = sum(graph[path[j]][path[j+1]].get('distance_km', 0) for j in range(len(path)-1))
                        full_distance += segment_distance
                        current_point = stop
                
                path, cost = (dijkstra(graph, current_point, v['end']) if algorithm == 'dijkstra'
                            else a_star(graph, current_point, v['end'], heuristic_distance))
                
                if path and path[1:]: 
                    full_path.extend(path[1:])
                    full_cost += cost
                    segment_distance = sum(graph[path[j]][path[j+1]].get('distance_km', 0) for j in range(len(path)-1))
                    full_distance += segment_distance
                
                routes[idx] = {
                    'vehicle': vehicle_idx,
                    'start': v['start'],
                    'end': v['end'],
                    'stops': v.get('additional_stops', []),
                    'path': full_path,
                    'cost': full_cost,
                    'distance': full_distance,
                    'weight': v['weight'],
                    'capacity': v['capacity'],
                    'weight_distribution': weight_distribution.get(vehicle_idx, {})
                }
            else:
                path, cost = (dijkstra(graph, v['start'], v['end']) if algorithm == 'dijkstra'
                            else a_star(graph, v['start'], v['end'], heuristic_distance))
                
                if path:
                    distance = sum(graph[path[j]][path[j+1]].get('distance_km', 0) for j in range(len(path)-1))
                    routes[idx] = {
                        'vehicle': vehicle_idx,
                        'start': v['start'],
                        'end': v['end'],
                        'path': path,
                        'cost': cost,
                        'distance': distance,
                        'weight': v['weight'],
                        'capacity': v['capacity'],
                        'weight_distribution': weight_distribution.get(vehicle_idx, {})
                    }
    
    return routes

# ===== Helper Functions =====
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth."""
    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Earth radius in kilometers
    radius = 6371.0
    
    # Calculate the distance
    distance = radius * c
    return distance

def interpolate_points_along_path(coord1, coord2, num_points=20):
    """Generate interpolated points along a straight line between two coordinates."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Create evenly spaced points
    lat_points = np.linspace(lat1, lat2, num_points)
    lon_points = np.linspace(lon1, lon2, num_points)
    
    # Combine into coordinate pairs
    return list(zip(lat_points, lon_points))

def create_detailed_vehicle_animation_geojson(routes):
    """Create a GeoJSON with detailed animation points interpolated along the edges of the route."""
    features = []
    
    # Set base time and animation parameters
    base_time = datetime.datetime(2025, 4, 27, 9, 0, 0)  # Starting at 9 AM
    
    # Colors for different routes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black', 'pink']
    vehicle_icons = ['truck', 'shipping', 'truck', 'car', 'truck', 'plane', 'truck', 'car', 'truck', 'shipping']
    
    for idx, route in routes.items():
        color = colors[idx % len(colors)]
        icon = vehicle_icons[idx % len(vehicle_icons)]
        path_nodes = route['path']
        
        if not path_nodes or len(path_nodes) < 2:
            continue
            
        # Extract coordinates for each node in the path
        node_coords = []
        for node in path_nodes:
            if pd.notna(G.nodes[node]['latitude']) and pd.notna(G.nodes[node]['longitude']):
                node_coords.append((G.nodes[node]['latitude'], G.nodes[node]['longitude']))
        
        if len(node_coords) < 2:
            continue
        
        # Create interpolated points along each edge
        all_coords = []
        edge_times = []
        edge_distances = []
        
        for i in range(len(node_coords) - 1):
            from_node = path_nodes[i]
            to_node = path_nodes[i+1]
            
            # Get edge weight (time in hours)
            edge_weight = G[from_node][to_node]['weight']
            
            # Get edge distance
            edge_distance = G[from_node][to_node].get('distance_km', 
                                                       haversine_distance(node_coords[i][0], node_coords[i][1],
                                                                          node_coords[i+1][0], node_coords[i+1][1]))
            
            # Number of points to generate along the edge (proportional to distance)
            points_per_km = 10  # 10 points per kilometer
            num_points = max(5, int(edge_distance * points_per_km))
            
            # Generate points along this edge
            points = interpolate_points_along_path(node_coords[i], node_coords[i+1], num_points)
            
            # Add all except the last point (to avoid duplicates)
            if i < len(node_coords) - 2:
                all_coords.extend(points[:-1])
            else:
                all_coords.extend(points)  # Include last point for final edge
            
            edge_times.append(edge_weight)
            edge_distances.append(edge_distance)
        
        # Calculate time intervals between each interpolated point
        total_points = len(all_coords)
        total_time = sum(edge_times)  # Total time in hours
        
        # Adjust total time based on animation speed (1-10 scale)
        animation_speed = st.session_state.get('animation_speed', 5)
        scaling_factor = 11 - animation_speed  # Invert scale: 10 -> 1, 1 -> 10
        adjusted_time = total_time * (scaling_factor / 5)  # Scale around the middle value of 5
        
        time_per_point = adjusted_time / total_points  # Time in hours per point
        
        current_time = base_time
        vehicle_label = f"Kendaraan {route['vehicle']+1}"
        weight_info = f"{route['weight']:.1f}/{route['capacity']:.1f} ton"
        
        # Create feature for each interpolated point with appropriate timing
        for i, coord in enumerate(all_coords):
            popup_content = f"{vehicle_label}<br>Muatan: {weight_info}"
            
            # Add information about current road segment
            current_node_idx = 0
            for j, edge_dist in enumerate(edge_distances):
                if current_node_idx + (edge_dist * points_per_km) > i:
                    from_label = id_to_label[path_nodes[j]]
                    to_label = id_to_label[path_nodes[j+1]]
                    popup_content += f"<br>Ruas: {from_label} → {to_label}"
                    break
                current_node_idx += int(edge_dist * points_per_km)
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [coord[1], coord[0]]  # GeoJSON uses [lon, lat]
                },
                'properties': {
                    'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'icon': icon,
                    'iconColor': color,
                    'iconSize': [24, 24],
                    'popup': popup_content,
                    'tooltip': f"{vehicle_label}"
                }
            }
            features.append(feature)
            
            # Increment time for next point (convert hours to seconds)
            time_increment = datetime.timedelta(seconds=time_per_point * 3600)
            current_time += time_increment
    
    return features

def add_smooth_vehicle_animation(m, routes):
    """Add smooth vehicle animation to the map using interpolated points along edges."""
    import json
    from folium.plugins import TimestampedGeoJson
    
    # Generate detailed animation features
    features = create_detailed_vehicle_animation_geojson(routes)
    
    # Add TimestampedGeoJson to the map
    TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period='PT1S',  # Update every 1 second
        duration='PT1S',  # Each point lasts for 1 second
        auto_play=True,
        loop=False,
        max_speed=30,  # Allow faster playback
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True,
        add_last_point=True
    ).add_to(m)
    
    return m

# ===== Streamlit App =====
st.set_page_config(layout="wide")
st.title("🚚 Sistem Rute Pengiriman Barang - Tebet, Jakarta Selatan")

st.sidebar.header("🛠️ Panel Kontrol")

depot_nodes_df = nodes_df[nodes_df['name'].str.lower().str.contains("depot")]
ruko_gedung_nodes_df = nodes_df[nodes_df['name'].str.lower().str.contains("ruko|gedung")]

max_vehicles = 15
num_vehicles = st.sidebar.slider("Jumlah Kendaraan", min_value=1, max_value=max_vehicles, value=1)

vehicle_data = []
st.sidebar.subheader("🚗 Detail Kendaraan")
for i in range(num_vehicles):
    with st.sidebar.expander(f"Kendaraan {i+1}"):
        start = st.selectbox(f"Start (Depot) - Kendaraan {i+1}", depot_nodes_df['name'], key=f"start_{i}")
        end = st.selectbox(f"End (Ruko/Gedung) - Kendaraan {i+1}", ruko_gedung_nodes_df['name'], key=f"end_{i}")
        weight = st.number_input(f"Berat Pengiriman (Ton) - Kendaraan {i+1}", min_value=0.0, max_value=15.0, value=1.0, step=0.1, key=f"weight_{i}")
        priority = st.number_input(f"Prioritas - Kendaraan {i+1}", min_value=1, max_value=15, value=1, key=f"priority_{i}")
        capacity = st.number_input(f"Kapasitas Kendaraan (Ton) - Kendaraan {i+1}", min_value=0.0, max_value=15.0, value=2.0, step=0.1, key=f"capacity_{i}")
        vehicle_data.append({
            'start': label_to_id[start],
            'end': label_to_id[end],
            'weight': weight,
            'priority': priority,
            'capacity': capacity
        })

algorithm = st.sidebar.selectbox("Algoritma", ['dijkstra', 'a_star'])

animation_speed = st.sidebar.slider("Kecepatan Animasi", 
                               min_value=1, 
                               max_value=10, 
                               value=5, 
                               help="Sesuaikan kecepatan animasi kendaraan")

# Store animation speed in session state
st.session_state['animation_speed'] = animation_speed

show_performance = st.sidebar.checkbox("Visualisasi Performa Komputasi")
visualization_option = st.sidebar.selectbox("Jenis Visualisasi", ["Peta dengan Animasi Halus", "Peta Folium Statis", "Graph (NetworkX)"])

# ===== Performance Measurement Dinamis =====
start_time = time.time()
tracemalloc.start()

routes = multi_vehicle_routing(G, vehicle_data, algorithm)

current, peak = tracemalloc.get_traced_memory()
end_time = time.time()
tracemalloc.stop()

execution_time = (end_time - start_time)
memory_usage = peak / 10**6

if visualization_option == "Peta dengan Animasi Halus":
    st.subheader("🗺️ Peta Visualisasi Rute dengan Animasi Halus Kendaraan")
    
    m = folium.Map(location=[-6.2297, 106.8532], zoom_start=15)
    
    # Add all nodes as markers
    for node in G.nodes:
        lat, lon = G.nodes[node].get('latitude'), G.nodes[node].get('longitude')
        label = G.nodes[node].get('label', str(node))
        if pd.notna(lat) and pd.notna(lon):
            Marker([lat, lon], tooltip=label, icon=folium.Icon(color="gray", icon="circle", prefix='fa')).add_to(m)
    
    # Draw routes with AntPath for better visual effect
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black', 'pink']
    for idx, route in routes.items():
        coords = [[G.nodes[n]['latitude'], G.nodes[n]['longitude']] for n in route['path'] if pd.notna(G.nodes[n]['latitude'])]
        if coords:
            # Use AntPath for animated path effect
            AntPath(
                locations=coords,
                color=colors[idx % len(colors)],
                weight=4,
                opacity=0.8,
                delay=1000,
                dash_array=[10, 20],
                tooltip=f"Rute Kendaraan {route['vehicle']+1}"
            ).add_to(m)
            
            # Add start and end markers
            start_label = id_to_label.get(route['start'], str(route['start']))
            end_label = id_to_label.get(route['end'], str(route['end']))
            
            folium.Marker(
                coords[0], 
                icon=folium.Icon(color="green", icon="play", prefix='fa'),
                tooltip=f"Start: {start_label}"
            ).add_to(m)
            
            folium.Marker(
                coords[-1], 
                icon=folium.Icon(color="red", icon="flag-checkered", prefix='fa'),
                tooltip=f"End: {end_label}"
            ).add_to(m)
            
            # Add markers for stops
            if 'stops' in route and route['stops']:
                for stop in route['stops']:
                    stop_lat = G.nodes[stop]['latitude']
                    stop_lon = G.nodes[stop]['longitude']
                    stop_label = id_to_label.get(stop, str(stop))
                    if pd.notna(stop_lat) and pd.notna(stop_lon):
                        folium.Marker(
                            [stop_lat, stop_lon], 
                            icon=folium.Icon(color="orange", icon="stop", prefix='fa'),
                            tooltip=f"Stop: {stop_label}"
                        ).add_to(m)
    
    # Add smooth vehicle animation along edges
    m = add_smooth_vehicle_animation(m, routes)
    
    # Display the map
    st_data = st_folium(m, width=900, height=500)
    
    
elif visualization_option == "Peta Folium Statis":
    st.subheader("🗺️ Peta Visualisasi Rute")
    m = folium.Map(location=[-6.2297, 106.8532], zoom_start=15)

    for node in G.nodes:
        lat, lon = G.nodes[node].get('latitude'), G.nodes[node].get('longitude')
        label = G.nodes[node].get('label', str(node))
        if pd.notna(lat) and pd.notna(lon):
            Marker([lat, lon], tooltip=label, icon=folium.Icon(color="gray", icon="circle")).add_to(m)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black', 'pink']
    for idx, route in routes.items():
        coords = [[G.nodes[n]['latitude'], G.nodes[n]['longitude']] for n in route['path'] if pd.notna(G.nodes[n]['latitude'])]
        if coords:
            folium.PolyLine(coords, color=colors[idx % len(colors)], weight=5,
                            tooltip=f"Rute Kendaraan {route['vehicle']+1}").add_to(m)

            start_label = id_to_label.get(route['start'], str(route['start']))
            end_label = id_to_label.get(route['end'], str(route['end']))

            folium.Marker(coords[0], icon=folium.Icon(color="green"),
                         tooltip=f"Start ({start_label})").add_to(m)
            folium.Marker(coords[-1], icon=folium.Icon(color="red"),
                         tooltip=f"End ({end_label})").add_to(m)
            
            if 'stops' in route and route['stops']:
                for stop in route['stops']:
                    stop_lat = G.nodes[stop]['latitude']
                    stop_lon = G.nodes[stop]['longitude']
                    stop_label = id_to_label.get(stop, str(stop))
                    if pd.notna(stop_lat) and pd.notna(stop_lon):
                        folium.Marker([stop_lat, stop_lon], 
                                     icon=folium.Icon(color="orange", icon="flag"),
                                     tooltip=f"Stop ({stop_label})").add_to(m)

    st_data = st_folium(m, width=900, height=500)

elif visualization_option == "Graph (NetworkX)":
    st.subheader("🔗 Visualisasi Graph Jaringan Jalan")
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) for node in G.nodes 
           if pd.notna(G.nodes[node].get('latitude')) and pd.notna(G.nodes[node].get('longitude'))}

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue', ax=ax)
    nx.draw_networkx_edges(G, pos, width=1, edge_color='lightgray', alpha=0.7, ax=ax)

    labels = {node: G.nodes[node]['label'] for node in G.nodes if 'label' in G.nodes[node]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    edge_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black', 'pink']
    for idx, route in routes.items():
        path_edges = list(zip(route['path'][:-1], route['path'][1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color=edge_colors[idx % len(edge_colors)], ax=ax)

    plt.title("Graph Transportation Network")
    plt.axis('off')
    st.pyplot(fig)


# ===== Dashboard Performa & Biaya =====
st.subheader("📊 Ringkasan Rute & Biaya per Kendaraan")

for idx, (key, route) in enumerate(routes.items(), start=1):
    s_label = id_to_label.get(route['start'], str(route['start']))
    e_label = id_to_label.get(route['end'], str(route['end']))
    path_labels = [id_to_label.get(n, str(n)) for n in route['path']]
    route_str = " ➔ ".join(path_labels)

    st.markdown(f"### 🚚 Kendaraan {route['vehicle']+1} - Rute #{idx}")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**📍 Asal:** {s_label}")
        st.write(f"**🎯 Tujuan Akhir:** {e_label}")
        
        if 'stops' in route and route['stops']:
            stops_str = ", ".join([id_to_label.get(stop, str(stop)) for stop in route['stops']])
            st.write(f"**🛑 Pemberhentian Tambahan:** {stops_str}")
        
        st.write(f"**🚛 Berat Muatan:** {route['weight']:.2f} ton / {route['capacity']:.2f} ton")

        st.write("**📦 Distribusi Muatan:**")
        if 'weight_distribution' in route and 'weights' in route['weight_distribution']:
            for dest, w in route['weight_distribution']['weights'].items():
                dest_label = id_to_label.get(dest, str(dest))
                st.markdown(f"- {dest_label}: **{w:.2f} ton**")
        else:
            st.markdown(f"- {e_label}: **{route['weight']:.2f} ton**")

    with col2:
        st.write("**🛣️ Rute Lengkap:**")
        st.markdown(route_str)

        st.write(f"**📏 Total Jarak:** {route['distance']:.2f} km")
        st.write(f"**⏱️ Estimasi Waktu:** {route['cost'] * 60:.0f} menit")
        
        biaya_per_km = 5000  # Rp per km
        total_biaya = route['distance'] * biaya_per_km
        st.write(f"**💸 Estimasi Biaya:** Rp {total_biaya:,.0f}")

    st.markdown("---")

# ===== Grafik Analisis Kinerja =====
if show_performance:
    st.subheader(f"⏱️ Analisis Kinerja Algoritma: {algorithm.upper()}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Waktu Komputasi", value=f"{execution_time:.4f} detik")

    with col2:
        st.metric(label="Penggunaan Memori", value=f"{memory_usage:.4f} MB")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].bar(['Waktu Komputasi'], [execution_time], color='skyblue')
    ax[0].set_title('Waktu Komputasi (detik)')
    ax[0].set_ylabel('Detik')

    ax[1].bar(['Penggunaan Memori'], [memory_usage], color='lightgreen')
    ax[1].set_title('Penggunaan Memori (MB)')
    ax[1].set_ylabel('MB')
    st.pyplot(fig)
