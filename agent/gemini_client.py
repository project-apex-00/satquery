import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

genai.configure(api_key=API_KEY)

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = [PRIMARY_MODEL, "gemini-3.5-flash-lite", "gemini-3.6-flash"]


def _call_gemini(prompt: str, fallback_text: str) -> str:
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            if resp and resp.text:
                return resp.text.strip()
        except Exception:
            continue
    return fallback_text


def ask_gemini(user_question: str, specialist_result: dict) -> str:
    pred_class = specialist_result.get("predicted_class", "Unknown")
    confidence = specialist_result.get("confidence", 0)
    all_probs = specialist_result.get("all_probs", {})

    prompt = f"""You are an expert remote-sensing intelligence assistant.

A fine-tuned remote-sensing specialist model (trained on Sentinel-2 satellite imagery)
analyzed the user's uploaded image and output this structured ground truth:

Predicted Land-Cover Class: {pred_class}
Confidence Score: {confidence} ({round(float(confidence)*100, 1)}%)
All Class Probabilities: {all_probs}

User Question: "{user_question}"

INSTRUCTIONS:
1. Using ONLY the specialist model's output above as factual ground truth, answer the user's question clearly.
2. Mention the primary land-cover type and the confidence level.
3. If the user asks about something beyond the classifier's capabilities, state what is observed from the model rather than hallucinating.
"""
    fallback = (
        f"The satellite image is classified as **{pred_class}** with **{round(float(confidence)*100, 1)}% confidence**. "
        f"Key probabilities: {list(all_probs.items())[:3]}."
    )
    return _call_gemini(prompt, fallback)


def ask_gemini_grounding(user_question: str, grounding_result: dict, classifier_result: dict = None) -> str:
    target_class = grounding_result.get("target_class", "Feature")
    coverage = grounding_result.get("coverage_percentage", 0)
    location = grounding_result.get("spatial_location", "central region")
    bbox = grounding_result.get("bounding_box", {})

    prompt = f"""You are a remote-sensing geospatial grounding specialist.

The user asked: "{user_question}"

The spatial grounding specialist tool analyzed the imagery and detected:
Target Feature Identified: {target_class}
Spatial Coverage: {coverage}% of total image surface
Location in Image: {location}
Bounding Box Coordinates: [ymin: {bbox.get('ymin')}, xmin: {bbox.get('xmin')}, ymax: {bbox.get('ymax')}, xmax: {bbox.get('xmax')}]

INSTRUCTIONS:
Explain to the user exactly where the feature is located in the image, its area percentage, and describe the visual bounding box highlighted on the evidence map.
"""
    fallback = (
        f"Located **{target_class}** spanning approximately **{coverage}%** of the image, concentrated in the **{location}**. "
        f"A spatial bounding box has been highlighted on the visual evidence map."
    )
    return _call_gemini(prompt, fallback)


def ask_gemini_change(user_question: str, change_result: dict, t1_result: dict = None, t2_result: dict = None) -> str:
    dominant = change_result.get("dominant_trend", "Surface Alteration")
    total_change = change_result.get("total_change_percentage", 0)
    veg_loss = change_result.get("vegetation_loss_percentage", 0)
    veg_gain = change_result.get("vegetation_gain_percentage", 0)
    builtup_gain = change_result.get("built_up_gain_percentage", 0)
    confidence = change_result.get("confidence", 0)

    prompt = f"""You are an Earth Observation change-detection specialist.

A bi-temporal change analysis was performed between two co-registered satellite images (T1 Before, T2 After):

Quantitative Change Metrics:
- Dominant Dynamic: {dominant}
- Total Surface Changed: {total_change}%
- Vegetation Reduction / Loss: {veg_loss}%
- Built-up / Urban Expansion: {builtup_gain}%
- Vegetation Regrowth: {veg_gain}%
- Detection Confidence: {round(float(confidence)*100, 1)}%

User Question: "{user_question}"

INSTRUCTIONS:
1. Directly answer whether the requested land feature increased, decreased, or remained unchanged.
2. Quote the specific change percentages from the metrics above as numerical evidence.
3. Explain the visual evidence displayed in the spatial change heatmap (e.g. Red for vegetation loss, Amber for built-up gain).
"""
    fallback = (
        f"Bi-temporal change analysis reveals **{dominant}** with **{total_change}% total surface alteration**. "
        f"Built-up area expanded by **{builtup_gain}%**, while vegetation shifted by **-{veg_loss}%**. "
        f"Review the colored change heatmap for the spatial distribution."
    )
    return _call_gemini(prompt, fallback)


def ask_gemini_fusion(user_question: str, fusion_result: dict) -> str:
    builtup = fusion_result.get("built_up_coverage_percentage", 0)
    water = fusion_result.get("water_coverage_percentage", 0)
    veg = fusion_result.get("vegetation_coverage_percentage", 0)
    clouds = fusion_result.get("optical_cloud_coverage_percentage", 0)
    penetrated = fusion_result.get("radar_cloud_penetration_percentage", 0)

    prompt = f"""You are a multi-sensor remote-sensing scientist specializing in Optical-SAR cross-modal fusion.

A co-registered Optical multispectral image and Synthetic Aperture Radar (SAR) backscatter image were fused:

Cross-Modal Metrics:
- Built-up Surface (Confirmed via SAR double-bounce): {builtup}%
- Water Bodies (Confirmed via low specular backscatter + absorption): {water}%
- Vegetation (Chlorophyll absorption + volume scattering): {veg}%
- Optical Cloud / Haze Coverage: {clouds}%
- Surface Features Penetrated by SAR through Cloud Cover: {penetrated}%

User Question: "{user_question}"

INSTRUCTIONS:
1. Answer how the optical and SAR channels complement each other to identify built-up structures and water bodies.
2. Mention the radar microwave's capability to penetrate optical cloud/atmospheric obstruction.
3. Cite the exact extracted percentages as ground truth.
"""
    fallback = (
        f"Cross-modal Optical-SAR fusion identified **{builtup}% built-up structures** (via SAR double-bounce) "
        f"and **{water}% water coverage** (via specular radar absorption). "
        f"SAR microwave successfully penetrated **{penetrated}%** of cloud/haze cover to reveal underlying ground features."
    )
    return _call_gemini(prompt, fallback)
