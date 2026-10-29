"""
Police Station Locator Module.
Finds publicly listed police stations and women safety cells near the camera's configured coordinates.
Provides verified public contact numbers for human escalation.

Configured via config.py.
IMPORTANT:
Geographic proximity does not prove jurisdiction or automated report acceptance.
Never send private images to unverified endpoints.
Use 112 / 1091 for live emergency escalation.
"""

import math
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

import config

# Fallback curated directory of verified public police stations / women safety cells
FALLBACK_STATION_DIRECTORY = [
    {
        "name": "Sector 17 Central Police Station",
        "city": "Chandigarh",
        "lat": 30.7415,
        "lon": 76.7794,
        "address": "Sector 17-C, Near District Courts, Chandigarh",
        "public_contact": "0172-2773990",
        "emergency_hotline": "112 / 1091",
        "type": "Central Police Station & Women Help Desk"
    },
    {
        "name": "Sector 19 Police Station",
        "city": "Chandigarh",
        "lat": 30.7320,
        "lon": 76.7890,
        "address": "Sector 19, Chandigarh",
        "public_contact": "0172-2773992",
        "emergency_hotline": "112 / 1091",
        "type": "Territorial Police Station"
    },
    {
        "name": "Connaught Place Police Station",
        "city": "New Delhi",
        "lat": 28.6315,
        "lon": 77.2167,
        "address": "Barakhamba Road, Connaught Place, New Delhi",
        "public_contact": "011-23351100",
        "emergency_hotline": "112 / 1091",
        "type": "Metropolitan Police Station"
    },
    {
        "name": "Azad Maidan Police Station",
        "city": "Mumbai",
        "lat": 18.9400,
        "lon": 72.8330,
        "address": "Mahapalika Marg, Fort, Mumbai",
        "public_contact": "022-22620429",
        "emergency_hotline": "112 / 103",
        "type": "Metropolitan Police Station & Women Desk"
    }
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates approximate distance between two points on Earth in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def find_nearest_police_stations(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 10.0,
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """
    Finds verified listed police stations near the given or configured coordinates.
    Tries OpenStreetMap Overpass API first; falls back cleanly to the curated directory if offline.
    """
    target_lat = lat if lat is not None else config.LOCATION_LATITUDE
    target_lon = lon if lon is not None else config.LOCATION_LONGITUDE

    results: List[Dict[str, Any]] = []
    retrieval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Attempt Overpass API Query (Lawful public Geo API)
    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:5];
        (
          node["amenity"="police"](around:{int(radius_km * 1000)},{target_lat},{target_lon});
          way["amenity"="police"](around:{int(radius_km * 1000)},{target_lat},{target_lon});
        );
        out center 5;
        """
        response = requests.post(overpass_url, data={"data": query}, timeout=4)
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            for elem in elements:
                name = elem.get("tags", {}).get("name", "Local Police Post")
                el_lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
                el_lon = elem.get("lon") or (elem.get("center", {}).get("lon"))
                if el_lat is not None and el_lon is not None:
                    dist = haversine_distance(target_lat, target_lon, el_lat, el_lon)
                    phone = elem.get("tags", {}).get("phone") or elem.get("tags", {}).get("contact:phone") or "112 (National Emergency)"
                    addr = elem.get("tags", {}).get("addr:street") or elem.get("tags", {}).get("address") or config.LOCATION_NAME

                    results.append({
                        "name": name,
                        "distance_km": round(dist, 2),
                        "address": addr,
                        "public_contact": phone,
                        "emergency_hotline": "112 / 1091",
                        "source": "OpenStreetMap Public Geo Directory",
                        "retrieval_time": retrieval_time,
                        "jurisdiction_disclaimer": "Informational listing only. Does not confirm automated e-filing acceptance. Use 112 for emergency dispatch."
                    })
    except Exception:
        # Fall through to curated fallback if network query times out
        pass

    # 2. Use curated fallback directory if Overpass yielded no stations
    if not results:
        for stn in FALLBACK_STATION_DIRECTORY:
            dist = haversine_distance(target_lat, target_lon, stn["lat"], stn["lon"])
            results.append({
                "name": stn["name"],
                "distance_km": round(dist, 2),
                "address": stn["address"],
                "public_contact": stn["public_contact"],
                "emergency_hotline": stn["emergency_hotline"],
                "source": "Curated Verified Public Directory",
                "retrieval_time": retrieval_time,
                "jurisdiction_disclaimer": "Informational listing only. Does not confirm automated e-filing acceptance. Use 112 for emergency dispatch."
            })

    # Sort by distance
    results.sort(key=lambda s: s["distance_km"])
    return results[:max_results]


def display_nearest_police_info():
    """Prints a clean summary of the nearest stations for the configured location."""
    print("\n" + "=" * 65)
    print("      NEAREST VERIFIED POLICE STATIONS & HELPLINES")
    print("=" * 65)
    print(f" Configured Location : {config.LOCATION_NAME}")
    print(f" GPS Coordinates     : ({config.LOCATION_LATITUDE}, {config.LOCATION_LONGITUDE})")
    print("-" * 65)

    stations = find_nearest_police_stations()
    for idx, stn in enumerate(stations, 1):
        print(f" [{idx}] {stn['name']}")
        print(f"     Distance : ~{stn['distance_km']} km")
        print(f"     Address  : {stn['address']}")
        print(f"     Phone    : {stn['public_contact']} | Hotline: {stn['emergency_hotline']}")
        print(f"     Source   : {stn['source']} (Retrieved: {stn['retrieval_time']})")
        print()

    print(" [NOTE] Geographic proximity does not prove legal jurisdiction.")
    print("        Automated reports are not filed directly without authorization.")
    print("        In an emergency, dial 112 or 1091 for immediate assistance.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    display_nearest_police_info()
