from flask import Flask, request, jsonify
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import re
import traceback
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

EXPECTED_API_KEY = "yap23003"

class UniversalVehicleRecommender:
    def __init__(self, json_file):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.data = self.load_data(json_file)
        self.repair_data = self.process_repair_orders()
        self.concern_embeddings = self.model.encode(
            [entry['normalized_concern'] for entry in self.repair_data]
        ) if self.repair_data else None

    def load_data(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_repair_orders(self):
        repair_data = []
        for entry in self.data:
            ro = entry.get('repair_order', {})
            vehicle_info = ro.get('vehicle_info', {})
            make = vehicle_info.get('make', 'unknown').lower().strip()
            model = vehicle_info.get('model', 'unknown').lower().strip()
            labor_time = float(ro.get('billing_info', {}).get('labor_time', 0))
            total_cost = float(ro.get('billing_info', {}).get('total_amount', 0))

            for concern in ro.get('customer_concerns', []):
                normalized_concern = concern.get('normalized_concern', '').lower().strip()
                if not normalized_concern:
                    continue

                for job in ro.get('jobs', []):
                    labor_job = job.get('description', '').strip().lower()
                    parts = [(p.get('name', '').strip().lower(), float(p.get('cost', 0.0))) for p in job.get('parts', []) if p.get('name')]

                    repair_data.append({
                        'normalized_concern': normalized_concern,
                        'labor_job': labor_job,
                        'parts': parts,
                        'make': make,
                        'model': model,
                        'labor_time': labor_time,
                        'total_cost': total_cost,
                        'invoice': ro.get('Invoice Number'),
                        'mileage': vehicle_info.get('mileage'),
                        'vin': vehicle_info.get('VIN', '').strip().lower(),
                        'license_plate': vehicle_info.get('License Plate', '').strip().lower(),
                        'raw_entry': ro
                    })
        return repair_data

    def get_recommendations(self, user_input, vehicle_make=None, vehicle_model=None):
        if not self.repair_data:
            return []

        user_embedding = self.model.encode([user_input.lower().strip()])[0]
        similarities = cosine_similarity([user_embedding], self.concern_embeddings)[0]

        labor_stats = defaultdict(lambda: {
            'count': 0, 'parts': defaultdict(lambda: {'count': 0, 'costs': []}), 'labor_times': []
        })

        for idx, entry in enumerate(self.repair_data):
            if (vehicle_make and entry['make'] != vehicle_make.lower()) or \
               (vehicle_model and entry['model'] != vehicle_model.lower()):
                continue

            if similarities[idx] > 0.4:
                labor = entry['labor_job']
                labor_stats[labor]['count'] += 1
                for part_name, part_cost in entry['parts']:
                    labor_stats[labor]['parts'][part_name]['count'] += 1
                    labor_stats[labor]['parts'][part_name]['costs'].append(part_cost)
                labor_stats[labor]['labor_times'].append(entry['labor_time'])

        sorted_labors = sorted(labor_stats.items(), key=lambda x: -x[1]['count'])[:3]

        results = []
        for labor, data in sorted_labors:
            total_cases = data['count']
            parts_with_relevance = []
            for part, part_data in data['parts'].items():
                relevance = part_data['count'] / total_cases
                avg_cost = np.mean(part_data['costs']) if part_data['costs'] else 0.0
                parts_with_relevance.append({
                    'name': part,
                    'confidence': round(relevance, 2),
                    'avg_cost': f"${avg_cost:.2f}"
                })

            top_parts = sorted(parts_with_relevance, key=lambda x: (-x['confidence'], -len(x['name'])))[:3]

            avg_time = np.mean(data['labor_times']) if data['labor_times'] else 0
            time_std = np.std(data['labor_times']) if data['labor_times'] else 0

            clean_labor_name = labor.replace('rr', 'Rear').replace('fr', 'Front').title()

            results.append({
                'labor_job': clean_labor_name,
                'case_count': total_cases,
                'parts': top_parts,
                'avg_labor_time': f"{avg_time:.1f} ± {time_std:.1f} hrs"
            })

        return results

    def get_history(self, vin=None, license_plate=None):
        vin = vin.strip().lower() if vin else None
        license_plate = license_plate.strip().lower() if license_plate else None

        history_dict = {}
        for entry in self.repair_data:
            if vin and entry['vin'] != vin:
                continue
            if license_plate and entry['license_plate'] != license_plate:
                continue

            invoice = entry['invoice']
            if invoice not in history_dict:
                history_dict[invoice] = {
                    "invoice_number": invoice,
                    "total_cost": entry["total_cost"],
                    "mileage": entry["mileage"],
                    "customer_concerns": set(),
                    "jobs": [],
                    "make": entry["make"],
                    "model": entry["model"],
                    "year": entry["raw_entry"].get("vehicle_info", {}).get("year"),
                    "vin": entry["vin"],
                    "license_plate": entry["license_plate"]
                }

            history_dict[invoice]["customer_concerns"].add(entry["normalized_concern"])
            history_dict[invoice]["jobs"].append({
                "description": entry["labor_job"],
                "parts": entry["parts"]
            })

        history = []
        for h in history_dict.values():
            h["customer_concerns"] = list(h["customer_concerns"])
            history.append(h)
        return history[:5]

recommender = UniversalVehicleRecommender("repair_orders_final.json")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        authorization = request.headers.get('Authorization')
        if not authorization or authorization != f"Bearer {EXPECTED_API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        point = data.get("point")
        if point == "ping":
            return jsonify({"result": "pong"})

        if point == "app.vehicle_recommendation.query":
            params = data.get("params", {})
            concern = params.get("concern")
            make = params.get("make", "").strip().lower()
            model = params.get("model", "").strip().lower()

            if not concern:
                return jsonify({"error": "Missing 'concern' field"}), 400

            vehicle_match_count = sum(
                1 for entry in recommender.repair_data
                if entry['make'] == make and entry['model'] == model
            )

            recommendations = recommender.get_recommendations(concern, make, model)
            return jsonify({
                "total_records": vehicle_match_count,
                "recommendations": recommendations
            })

        return jsonify({"error": "Invalid point"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['POST'])
def history():
    try:
        data = request.get_json()
        authorization = request.headers.get('Authorization')
        if not authorization or authorization != f"Bearer {EXPECTED_API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        point = data.get("point")
        if point == "ping":
            return jsonify({"result": "pong"})

        if point == "app.vehicle_history.query":
            params = data.get("params", {})
            vin = params.get("vin")
            license_plate = params.get("license_plate")

            if not vin and not license_plate:
                return jsonify({"error": "Missing VIN or license plate"}), 400

            history = recommender.get_history(vin, license_plate)
            return jsonify({"records": history})

        return jsonify({"error": "Invalid point"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
