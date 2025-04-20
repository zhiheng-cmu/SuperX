import os
import math
import re
import subprocess
import time

import requests
import polyline
from dotenv import load_dotenv
import hashlib
from PIL import Image
import io
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np
from geopy.distance import geodesic
import random
import folium
from config import LLM_PROVIDER
from model.gemini import GeminiModel
from model.gpt4v import GPT4VModel
from model.claude import ClaudeModel



# AIzaSyAiLHN0SsxnsYjj3ycy8jv12JUUbIPkBkw
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
street_view_resolution = "640x400"

map_dict = {"map1_downtown_bk": [[40.689981, -73.983547],
                                 [40.691480, -73.987365],
                                 [40.694969, -73.987166],
                                 [40.694847, -73.984001],
                                 [40.689973, -73.981591]],
            "map2_soho": [[40.725781, -74.000906],
                         [40.724238, -73.997790],
                         [40.721791, -73.999872],
                         [40.722941, -74.002176]],
            "map3_time_square": [[40.759789, -73.987923],
                                 [40.757398, -73.982247],
                                 [40.753548, -73.985047],
                                 [40.755979, -73.990729]],
            "map4_williamsburg": [[40.723086, -73.958733],
                                  [40.718807, -73.952433],
                                  [40.717360, -73.954000],
                                  [40.721422, -73.960540]],
            "map5_chinatown": [[40.719086, -73.996479],
                               [40.717375, -73.991236],
                               [40.715121, -73.992582],
                               [40.716862, -73.997730]]}




def area2grid(api_key, lat_lng_corners, grid_size, use_portion=0.2, max_num_waypoint=20):
    lat = lat_lng_corners[:, 0]
    lng = lat_lng_corners[:, 1]
    min_lat, max_lat = min(lat), max(lat)
    min_lng, max_lng = min(lng), max(lng)

    # Approximate meters to degrees conversion (varies with latitude)
    lat_step = geodesic(meters=grid_size).destination((min_lat, min_lng), 0)[0] - min_lat
    lng_step = geodesic(meters=grid_size).destination((min_lat, min_lng), 90)[1] - min_lng

    # Generate lat/lng grid
    lat_points = np.arange(min_lat, max_lat, lat_step)
    lng_points = np.arange(min_lng, max_lng, lng_step)

    # Create grid points
    grid_points = [(lat, lng) for lat in lat_points for lng in lng_points]
    # Check if Street View is available
    useful_grid_pts = []
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    for coord in tqdm(grid_points, desc=f"checking sampled grid points to keep only those available for streetview"):
        lat = coord[0]
        lng = coord[1]
        metadata_params = {
            "location": f"{lat},{lng}",
            "radius": 10,
            "source": "outdoor",
            "key": api_key
        }

        metadata_response = requests.get(metadata_url, params=metadata_params)
        if (metadata_response.status_code != 200 or
                metadata_response.json().get("status") != "OK"):
            # print(f"No Street View imagery at location: {lat}, {lng}")
            continue
        else:
            useful_grid_pts.append((lat, lng))

    # Google Maps allow max num of waypoints to be 25. we use 20 by default just to be conservative
    num_pts_wanted = int(len(useful_grid_pts) * use_portion)
    num_pts_wanted = min(max_num_waypoint, num_pts_wanted)
    print(f"{len(useful_grid_pts)} out of {len(grid_points)} grid points do have street view images, "
          f"using {num_pts_wanted} of it")
    route_pts = random.choices(useful_grid_pts, k=num_pts_wanted)

    return route_pts


def get_routes_from_pts(location_list, route_file_out="route_map.html", route_mode="driving"):
    all_positions_and_headings = []
    way_points = []

    for i in range(1, len(location_list) - 1):
        location = ", ".join(map(str, location_list[i]))
        way_points.append(location)
        # 1. Get route polyline
        # print(start, end)
    start = ", ".join(map(str, location_list[0]))
    end = ", ".join(map(str, location_list[-1]))
    print(f"route start & end: {start} | {end}, with total {len(location_list) - 2} way points in between")
    # print(f"route waypoints: {start} | {way_points} | {end}")

    overview_poly, distance, duration = get_directions(start, end, GOOGLE_MAPS_API_KEY, mode=route_mode,
                                                       way_points=way_points)

    ''' ============= Create map visualization ============= '''
    # Decode polyline
    decoded_pts = polyline.decode(overview_poly)
    route_map = folium.Map(location=decoded_pts[0], zoom_start=7)

    # Add polyline to the map
    folium.PolyLine(decoded_pts, color="blue", weight=5, opacity=0.7).add_to(route_map)

    # save the map
    route_map.save(route_file_out)

    # 3. Interpolate additional points
    sampled_points = []
    for i in range(len(decoded_pts) - 1):
        interpolated = interpolate_points(
            decoded_pts[i][0], decoded_pts[i][1],
            decoded_pts[i + 1][0], decoded_pts[i + 1][1]
        )
        sampled_points.extend(interpolated[:-1])
    sampled_points.append(decoded_pts[-1])

    # 4. Compute headings
    headings = []
    for i in range(len(sampled_points) - 1):
        lat1, lng1 = sampled_points[i]
        lat2, lng2 = sampled_points[i + 1]
        b = compute_bearing(lat1, lng1, lat2, lng2)
        headings.append(b)

    # 5. Build list of positions and headings with movement tracking
    for i in range(len(sampled_points) - 1):
        lat_cur, lng_cur = sampled_points[i]
        old_heading = headings[i]

        # Add straight movement
        all_positions_and_headings.append((lat_cur, lng_cur, old_heading, "forward"))

        # If there's a turn coming
        if i < len(headings) - 1:
            new_heading = headings[i + 1]
            turn_direction = determine_turn_direction(old_heading, new_heading)
            if turn_direction != "forward":
                turn_seq = generate_turn_headings(old_heading, new_heading, step=12)
                lat_turn, lng_turn = sampled_points[i + 1]
                for h in turn_seq[1:]:
                    all_positions_and_headings.append((lat_turn, lng_turn, h, turn_direction))

    return all_positions_and_headings, distance, duration


def get_routes_from_locations(location_list, route_file_output, route_mode="driving"):
    '''

    :param location_list: list of well formatted string locations that can be used directly by GoogleMaps for navigation
    :param map_name:
    :param route_file_output: e.g. f"{BASE_DATA_DIR}/{map_name}/route_map.html"
    :return:
    '''
    all_positions_and_headings = []
    way_points = []

    for i in range(1, len(location_list) - 1):
        location = location_list[i]
        way_points.append(location)

    start = location_list[0]
    end = location_list[-1]
    print(f"route start & end: {start} | {end}, with total {len(location_list)-2} way points in between")
    # print(f"route waypoints: {start} | {way_points} | {end}")

    route_json, overview_poly, distance, duration = get_directions(start, end, GOOGLE_MAPS_API_KEY, mode=route_mode, way_points=way_points)

    ''' ============= Create map visualization ============= '''
    # Decode polyline
    decoded_pts = polyline.decode(overview_poly)
    route_map = folium.Map(location=decoded_pts[0], zoom_start=7)

    # Add polyline to the map
    folium.PolyLine(decoded_pts, color="blue", weight=5, opacity=0.7).add_to(route_map)

    # save the map
    route_map.save(route_file_output)

    # 3. Interpolate additional points
    sampled_points = []
    for i in range(len(decoded_pts) - 1):
        interpolated = interpolate_points(
            decoded_pts[i][0], decoded_pts[i][1],
            decoded_pts[i + 1][0], decoded_pts[i + 1][1]
        )
        sampled_points.extend(interpolated[:-1])
    sampled_points.append(decoded_pts[-1])

    # 4. Compute headings
    headings = []
    for i in range(len(sampled_points) - 1):
        lat1, lng1 = sampled_points[i]
        lat2, lng2 = sampled_points[i + 1]
        b = compute_bearing(lat1, lng1, lat2, lng2)
        headings.append(b)

    # 5. Build list of positions and headings with movement tracking
    for i in range(len(sampled_points) - 1):
        lat_cur, lng_cur = sampled_points[i]
        old_heading = headings[i]

        # Add straight movement
        all_positions_and_headings.append((lat_cur, lng_cur, old_heading, "forward"))

        # If there's a turn coming
        if i < len(headings) - 1:
            new_heading = headings[i + 1]
            turn_direction = determine_turn_direction(old_heading, new_heading)
            if turn_direction != "forward":
                turn_seq = generate_turn_headings(old_heading, new_heading, step=12)
                lat_turn, lng_turn = sampled_points[i + 1]
                for h in turn_seq[1:]:
                    all_positions_and_headings.append((lat_turn, lng_turn, h, turn_direction))

    return all_positions_and_headings, distance, duration


def determine_turn_direction(old_heading, new_heading):
    """Determine if the turn is left or right"""
    diff = ((new_heading - old_heading + 180) % 360) - 180
    if abs(diff) < 10:  # threshold for forward movement
        return "forward"
    return "right" if diff > 0 else "left"


def get_directions(origin, destination, api_key, mode="driving", way_points=None):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    if way_points is not None:
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "waypoints": "|".join(way_points),
            "key": api_key
        }
    else:
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": api_key
        }
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] != "OK":
        raise ValueError(f"Directions API error: {data.get('error_message', data['status'])}")
    route = data["routes"][0]
    distance = route["legs"][0]["distance"]["text"]
    duration = route["legs"][0]["duration"]["text"]
    return route, route["overview_polyline"]["points"], distance, duration


def get_coord(location, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": location,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] == "OK":
        coordinates = data["results"][0]["geometry"]["location"]
        lat = coordinates["lat"]
        lng = coordinates["lng"]
        print(f"Latitude: {lat}, Longitude: {lng}")
        # wait for 1 second to avoid exceeding api limit request per second
        time.sleep(1)
        return [lat, lng]
    else:
        raise RuntimeError(f"Failed to get location coord: {location} {data.get('error_message', data['status'])}")


def normalize_coords(coords):
    """ Normalize a list of [lat, lng] coordinates to [x, y] in meters. """
    origin = coords[0]  # Take the first coordinate as the origin
    normalized_coords = [[0, 0]]  # Origin point at [0, 0]

    for coord in coords[1:]:
        lat = coord[0]
        lng = coord[1]
        # Calculate the x (longitude) distance
        x = geodesic((origin[0], origin[1]), (origin[0], lng)).meters
        # Adjust the sign based on longitude difference
        x *= -1 if lng < origin[1] else 1

        # Calculate the y (latitude) distance
        y = geodesic((origin[0], origin[1]), (lat, origin[1])).meters
        # Adjust the sign based on latitude difference
        y *= -1 if lat < origin[0] else 1

        normalized_coords.append([x, y])

    return normalized_coords

def compute_bearing(lat1, lng1, lat2, lng2):
    lat1_r, lng1_r = map(math.radians, [lat1, lng1])
    lat2_r, lng2_r = map(math.radians, [lat2, lng2])
    d_lng = lng2_r - lng1_r
    x = math.cos(lat2_r) * math.sin(d_lng)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lng))
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360


def interpolate_points(lat1, lng1, lat2, lng2, step=0.0001):
    """Interpolate points between two coordinates using a fixed step size."""
    points = []
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    distance = math.sqrt(dlat ** 2 + dlng ** 2)
    num_steps = max(1, int(distance / step))

    for i in range(num_steps + 1):
        fraction = i / num_steps
        lat = lat1 + fraction * dlat
        lng = lng1 + fraction * dlng
        points.append((lat, lng))
    return points


def generate_turn_headings(old_heading, new_heading, step=30):
    """Generate a series of intermediate heading angles between old_heading and new_heading."""
    turn_frames = []
    old_h = old_heading % 360
    new_h = new_heading % 360

    # Calculate the shortest turning direction
    diff = ((new_h - old_h + 180) % 360) - 180
    abs_diff = abs(diff)

    # Number of intermediate steps
    num_steps = int(abs_diff / step)

    for i in range(num_steps + 1):
        fraction = i / num_steps if num_steps > 0 else 1
        current_h = (old_h + diff * fraction) % 360
        turn_frames.append(current_h)

    # Ensure the final heading is included
    if not turn_frames or abs(turn_frames[-1] - new_h) > 0.1:
        turn_frames.append(new_h)

    return turn_frames


def generate_video_from_img(folder_dir, out_dir, frame_rate=5, reverse=True):
    if not os.path.exists(folder_dir) or len(os.listdir(folder_dir)) == 0:
        print(f"input image folder {folder_dir} is empty or does not exist")
        return False
    img_list = os.listdir(folder_dir)

    img_list = sorted(img_list, reverse=(not reverse))
    sample_img = cv2.imread(f"{folder_dir}/{img_list[0]}")
    sample_img = cv2.resize(sample_img, dsize=(sample_img.shape[1], sample_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    video_width = sample_img.shape[1]
    video_height = sample_img.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out0 = cv2.VideoWriter(out_dir, fourcc, frame_rate, (video_width, video_height))

    for i in tqdm(range(len(img_list)), desc=f"{out_dir}"):
        img_filename = img_list[i]
        img_dir = f"{folder_dir}/{img_filename}"
        img = cv2.imread(img_dir)
        img = cv2.resize(img, dsize=(sample_img.shape[1], sample_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        out0.write(img)
    return True


def get_image_hash(image_data):
    """Generate a hash of the image content."""
    return hashlib.md5(image_data).hexdigest()


def fetch_street_view_image(lat, lng, heading, api_key,
                            size=street_view_resolution, pitch=0, fov=120,
                            folder="frames", seen_hashes=None,
                            frame_index=0, surround_view=False):
    """
    Fetch and save a Street View image, labeled with a reversed frame_index.
    This ensures that when frames are sorted in descending order, the earliest
    route frames appear first in the final video.
    """
    if seen_hashes is None:
        seen_hashes = set()

    # Check if Street View is available
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    metadata_params = {
        "location": f"{lat},{lng}",
        "radius": 10,
        "source": "outdoor",
        "key": api_key
    }

    metadata_response = requests.get(metadata_url, params=metadata_params)
    if (metadata_response.status_code != 200 or
            metadata_response.json().get("status") != "OK"):
        print(f"No Street View imagery at location: {lat}, {lng}")
        return None

    # If imagery is available, fetch it
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    img_list = []
    if surround_view:
        heading_list = [heading-120, heading, heading+120]
        if not os.path.exists(f"{folder}/frames_surround"):
            os.makedirs(f"{folder}/frames_surround")
    else:
        heading_list = [heading]

    heading_name_list = ["left", "front", "right"]
    for i in range(len(heading_list)):
        heading_i = heading_list[i]
        heading_name = heading_name_list[i]
        if not os.path.exists(f"{folder}/frames_{heading_name}"):
            os.makedirs(f"{folder}/frames_{heading_name}")

        params = {
            "size": size,
            "location": f"{lat},{lng}",
            "heading": heading_i,
            "pitch": pitch,
            "fov": fov,
            "source": "outdoor",
            "key": api_key
        }

        response = requests.get(base_url, params=params, stream=True)
        if response.status_code == 200:
            image_data = response.content

            # Name the file using reverse index
            # Example: frame_0099_40.74844_-73.98401_0.0.jpg
            # The leading zeros in index ensure correct alphabetical sorting if needed
            filename = (f"{folder}/frames_{heading_name}/frame_{frame_index:05d}_"
                        f"{lat:.5f}_{lng:.5f}_{heading_i:.1f}.jpg")
            if surround_view:
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                img_list.append(image)
            with open(filename, 'wb') as f:
                f.write(image_data)
        else:
            print(f"Failed to fetch Street View: {response.status_code}")
            return None
    true_heading_filename = (f"{folder}/frames_surround/frame_{frame_index:05d}_"
                f"{lat:.5f}_{lng:.5f}_{heading:.1f}.jpg")

    if surround_view:
        surround_view_img = np.concatenate(img_list, axis=1)
        cv2.imwrite(true_heading_filename, surround_view_img)
    return true_heading_filename


def get_frame_number(filename):
    """Extract frame number from filename"""
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else 0


def get_model():
    """Get the appropriate model based on config"""
    if LLM_PROVIDER == "gemini":
        return GeminiModel()
    elif LLM_PROVIDER == "openai":
        return GPT4VModel()
    elif LLM_PROVIDER == "anthropic":
        return ClaudeModel()
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def analyze_street_view(image_directory, question_list, out_dir, surround=False, batches=None, total_length=-1):
    """Analyze street view images in a directory and generate spatial data"""
    # Get and sort image paths from frames directory
    if surround and os.path.exists(f"{image_directory}/frames_surround"):
        frames_dir = f"{image_directory}/frames_surround"
    else:
        frames_dir = f"{image_directory}/frames"

    if os.path.exists(frames_dir):
        image_directory = frames_dir

    # Get all image paths
    image_paths = []
    for image_file in os.listdir(image_directory):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(image_directory, image_file)
            image_paths.append(full_path)

    if not image_paths:
        print(f"No images found in {image_directory}!")
        return None

    # Sort by frame number in reverse order
    image_paths = sorted(image_paths, key=lambda x: get_frame_number(os.path.basename(x)), reverse=True)
    if total_length != -1:
        image_paths = image_paths[:total_length]
    print(f"\nFound {len(image_paths)} images in {image_directory}")

    try:
        # Initialize model based on config
        model = get_model()
        results = model.analyze_images(image_paths, question_list, out_dir, batches=batches)
        return results, image_paths

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None


def get_active_project_id():
    """Retrieve the active Google Cloud project ID from gcloud configuration."""
    try:
        # Run gcloud command to get active project ID
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error retrieving project ID: {e}")
        return None



def parse_json_from_response(response_text):
    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    data = []
    if json_match:
        json_data_str = json_match.group(1)
        data = json.loads(json_data_str)
    return data

def get_coord_from_route_data(route_data):
    result = []
    for frame in route_data["frames"]:
        coord = frame["coordinates"]
        lat = coord["lat"]
        lng = coord["lng"]
        result.append([lat, lng])

    return result


def coord_to_meter(coord, origin=[0, 0]):
    """
    Convert an individual [lat, lng] coordinate to [x, y] meters from the origin.

    Args:
        coord: The coordinate as [lat, lng].
        origin: The origin coordinate as [lat, lng].

    Returns:
        A list [x, y] representing the coordinate in meters from the origin.
    """
    # Calculate x distance (longitude difference)
    x = geodesic((origin[0], origin[1]), (origin[0], coord[1])).meters
    if coord[1] < origin[1]:
        x = -x  # Adjust sign for west

    # Calculate y distance (latitude difference)
    y = geodesic((origin[0], origin[1]), (coord[0], origin[1])).meters
    if coord[0] < origin[0]:
        y = -y  # Adjust sign for south

    return [x, y]


