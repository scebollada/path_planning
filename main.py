import sys

import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import yaml
import pandas as pd

#Function to plot the path
def plot_path(lat, long, origin_point, destination_point):
    """
    Given a list of latitudes and longitudes, origin
    and destination point, plots a path on a map

    Parameters
    ----------
    lat, long: list of latitudes and longitudes
    origin_point, destination_point: co-ordinates of origin
    and destination
    Returns
    -------
    Nothing. Only shows the map.
    """
    # adding the lines joining the nodes
    fig = go.Figure(go.Scattermapbox(
        name="Path",
        mode="lines",
        lon=long,
        lat=lat,
        marker={'size': 10},
        line=dict(width=4.5, color='blue')))
    # adding source marker
    fig.add_trace(go.Scattermapbox(
        name="Source",
        mode="markers",
        lon=[origin_point[1]],
        lat=[origin_point[0]],
        marker={'size': 12, 'color': "red"}))

    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name="Destination",
        mode="markers",
        lon=[destination_point[1]],
        lat=[destination_point[0]],
        marker={'size': 12, 'color': 'green'}))

    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="open-street-map", # carto-positron, open-street-map
                      mapbox_center_lat=30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox={
                          'center': {'lat': lat_center,
                                     'lon': long_center},
                          'zoom': 13})
    fig.show()

#Function to generate a list of lines that follow the path
def node_list_to_path(G, node_list):
    """
    Given a list of nodes, return a list of lines that together
    follow the path
    defined by the list of nodes.
    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    Returns
    -------
    lines : list of lines given as pairs ( (x_start, y_start),
    (x_stop, y_stop) )
    """
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(),
                   key=lambda x: x['length'])
        # if it has a geometry attribute
        if 'geometry' in data:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute,
            # then the edge is a straight line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    return lines


def main():
    # Getting data from config.yaml
    try:
        with open(r'config.YAML') as file:
            param_list = yaml.load(file, Loader=yaml.FullLoader)
            # print(param_list)
    except:
        print("YAML loading error!...")
    try:
        input_osm_file = param_list.get('input_osm_file')
        plot_route=bool(param_list.get('plot_route'))
        output_csv = bool(param_list.get('output_csv'))
        (orig_lat, orig_lon) = eval(param_list.get('orig'))
        (dest_lat, dest_lon) = eval(param_list.get('dest'))
        csv_folder=param_list.get('csv_folder')

    except:
        print("Error getting params from config.YAML!...")

    #Load the .OSM file (nodes and edges)
    filepath=input_osm_file
    G= ox.graph.graph_from_xml(filepath, bidirectional=True, simplify=False, retain_all=True)
    # Plotting the map graph
    # ox.plot_graph(G,bgcolor='white', node_color='blue',edge_color='gray')

    # Load the origin and desination locations in GPS coordinates
    origin_point=(orig_lat, orig_lon)
    destination_point=(dest_lat, dest_lon)

    # Get the nearest nodes to the locations
    origin_node=ox.distance.nearest_nodes(G, X=origin_point[1], Y=origin_point[0], return_dist=False)
    destination_node=ox.distance.nearest_nodes(G, X=destination_point[1], Y=destination_point[0], return_dist=False)

    print('Origin node: ')
    print(origin_node)
    print('Destination node: ')
    print(destination_node)


    # Finding the optimal path
    route = nx.shortest_path(G, origin_node, destination_node, weight='length') #by default, it uses dijkstra
    #Same result by using the dijkstra_path function
    # route = nx.dijkstra_path(G, origin_node, destination_node, weight='length')


    # Getting coordinates of the nodes
    lines = node_list_to_path(G, route)
    # we will store the longitudes and latitudes in following list
    long2 = []
    lat2 = []
    data=[] # to store the GPS coordinates in a .csv file
    for i in range(len(lines)):
        z = list(lines[i])
        l1 = list(list(zip(*z))[0])
        l2 = list(list(zip(*z))[1])
        for j in range(len(l1)):
            long2.append(l1[j])
            lat2.append(l2[j])
            data.append([lat2[j], long2[j]])


    # Plot the roue on the open street map
    if plot_route:
        plot_path(lat2, long2, origin_point, destination_point)

    #Save the GPs coordinates in a .csv file
    if output_csv:
        d = pd.DataFrame(data)
        file_name = csv_folder+'from_('+str(orig_lat)+' , '+str(orig_lon)+')_to_(' +str(dest_lat)+' , '+str(dest_lon)+')_.csv'
        d.to_csv(file_name, header=['latitude', 'longitude'], index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


