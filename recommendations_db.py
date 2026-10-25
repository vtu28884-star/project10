# recommendations_db.py

HEALTH_RECOMMENDATIONS = {
    0: {
        "condition": "Normal",
        "description": "All monitored vitals (HR, SpO2, Temperature) are within normal baseline ranges.",
        "natural_cure": {
            "foods_to_eat": [
                "Fresh seasonal fruits (Apples, Oranges)",
                "Hydrating vegetables (Cucumber, Zucchini)",
                "Whole grains and oats",
                "Minimum 2.5 - 3.0 Liters of water daily"
            ],
            "lifestyle": [
                "Maintain regular moderate physical activity (30 mins/day)",
                "Consistent 7-8 hours sleep schedule"
            ]
        },
        "medical_cure": {
            "medications": ["None required."],
            "advice": "All vitals stable. No medical intervention necessary."
        }
    },
    1: {
        "condition": "High Fever (Hyperthermia)",
        "description": "Elevated core body temperature detected (> 38.3 °C / 101 °F).",
        "natural_cure": {
            "foods_to_eat": [
                "Tender coconut water (replenishes electrolytes)",
                "Fresh warm ginger & tulsi infusion",
                "Clear vegetable broth or strained lentil soup",
                "Vitamin C rich citrus fruits (Oranges, Sweet Lime)"
            ],
            "lifestyle": [
                "Cold/tepid water sponge application on forehead and neck",
                "Wear light, breathable cotton clothing",
                "Complete physical rest in a well-ventilated room"
            ]
        },
        "medical_cure": {
            "medications": [
                "Paracetamol / Acetaminophen (500mg or 650mg as per age/weight every 6-8 hrs)",
                "Oral Rehydration Salts (ORS) solution"
            ],
            "advice": "Monitor temperature every 2 hours. If temperature exceeds 39.5 °C or persists beyond 48 hours, seek immediate clinical evaluation."
        }
    },
    2: {
        "condition": "Cardiovascular Stress (Tachycardia / Arrhythmia)",
        "description": "Heart rate exceeds standard demographic resting bounds (> 100 BPM at rest).",
        "natural_cure": {
            "foods_to_eat": [
                "Potassium-rich foods (Bananas, Coconut water, Avocado)",
                "Magnesium sources (Pumpkin seeds, Almonds, Dark leafy greens)",
                "Beetroot juice (promotes vasodilation via dietary nitrates)",
                "Herbal chamomile tea"
            ],
            "lifestyle": [
                "Engage in diaphragmatic breathing: 4-7-8 breathing technique",
                "Immediately cease caffeine, tobacco, energy drinks, and high-sodium foods",
                "Rest in a seated or semi-reclined position"
            ]
        },
        "medical_cure": {
            "medications": [
                "Consult physician for prescription Beta-blockers (e.g., Metoprolol/Atenolol) if chronically high"
            ],
            "advice": "Check blood pressure immediately. If accompanied by chest tightness, shortness of breath, or numbness radiating to the arm, contact emergency services."
        }
    },
    3: {
        "condition": "Hypoxia (Respiratory Stress)",
        "description": "Critically low peripheral capillary oxygen saturation detected (SpO2 < 92%).",
        "natural_cure": {
            "foods_to_eat": [
                "Iron-rich foods (Pomegranate, Spinach, Beetroot, Lentils)",
                "Antioxidant-dense berries and walnuts",
                "Warm warm water with honey and turmeric"
            ],
            "lifestyle": [
                "Adopt prone positioning (lie face down with pillows under chest and pelvis)",
                "Pursed-lip breathing exercises to maximize alveolar gas exchange",
                "Ensure cross-ventilation and open fresh airflow in the room"
            ]
        },
        "medical_cure": {
            "medications": [
                "Supplemental Oxygen therapy (target SpO2 >= 95%)",
                "Short-acting bronchodilator inhalers (e.g., Salbutamol/Albuterol) if respiratory wheezing is present"
            ],
            "advice": "URGENT: SpO2 below 90% is a clinical emergency. Arrange immediate hospital transfer for oxygen support."
        }
    }
}